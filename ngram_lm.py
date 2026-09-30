import math
from collections import defaultdict, Counter
from xml.parsers.expat import model

class NGramLanguageModel:
    def __init__(self, n=2, smoothing=False):
        """
        Khởi tạo mô hình N-gram Language Model.
        :param n: Độ dài N-gram (1: Unigram, 2: Bigram, 3: Trigram)
        :param smoothing: True để dùng Laplace Smoothing, False để dùng MLE
        """
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()
        self.total_tokens = 0

    def build_vocabulary(self, corpus):
        """
        Xây dựng tập từ vựng (Vocabulary) và đếm tổng số tokens.
        :param corpus: Danh sách các câu, mỗi câu là danh sách các token (list of list of str)
        """
        self.vocab = set()
        self.total_tokens = 0
        for sentence in corpus:
            for word in sentence:
                self.vocab.add(word)
                self.total_tokens += 1
        return self.vocab

    def _get_ngrams(self, sentence):
        """Tạo các n-gram từ một câu đã được padding ký tự bắt đầu/kết thúc."""
        if self.n > 1:
            padded_sentence = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        else:
            padded_sentence = sentence
            
        ngrams = []
        for i in range(len(padded_sentence) - self.n + 1):
            ngram = tuple(padded_sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams

    def fit(self, corpus):
        """
        Huấn luyện mô hình N-gram bằng cách đếm tần suất xuất hiện (counts).
        """
        self.build_vocabulary(corpus)
        
        # Bổ sung ký tự đặc biệt vào vocab nếu n > 1
        if self.n > 1:
            self.vocab.add('<s>')
            self.vocab.add('</s>')

        for sentence in corpus:
            ngrams = self._get_ngrams(sentence)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

    def probability(self, context, word):
        """
        Tính xác suất điều kiện P(word | context).
        :param context: Danh sách hoặc tuple các từ ngữ cảnh
        :param word: Từ tiếp theo
        """
        V = len(self.vocab)

        # 1. Unigram Model
        if self.n == 1:
            count_w = self.ngram_counts[(word,)]
            if self.smoothing:
                return (count_w + 1) / (self.total_tokens + V)
            return count_w / self.total_tokens if self.total_tokens > 0 else 0.0

        # 2. Bigram / Trigram Model
        context_tuple = tuple(context)
        ngram = context_tuple + (word,)
        
        count_ngram = self.ngram_counts[ngram]
        count_context = self.context_counts[context_tuple]

        if self.smoothing:
            return (count_ngram + 1) / (count_context + V)
        else:
            if count_context == 0:
                return 0.0
            return count_ngram / count_context

    def sentence_probability(self, sentence):
        """Tính xác suất tích của toàn bộ câu (chỉ phù hợp với câu rất ngắn)."""
        ngrams = self._get_ngrams(sentence)
        prob = 1.0
        for ngram in ngrams:
            context = ngram[:-1]
            word = ngram[-1]
            prob *= self.probability(context, word)
        return prob

    def sentence_log_probability(self, sentence):
        """
        Tính log probability của câu để tránh numerical underflow.
        log P(S) = sum(log P(w_t | context))
        """
        ngrams = self._get_ngrams(sentence)
        log_prob_sum = 0.0

        for ngram in ngrams:
            context = ngram[:-1]
            word = ngram[-1]
            p = self.probability(context, word)
            
            if p == 0.0:
                return float('-inf')  # Tránh lỗi math domain error khi log(0)
            
            log_prob_sum += math.log(p)

        return log_prob_sum

    def next_word_distribution(self, context):
        """
        Trả về phân phối xác suất của tất cả các từ trong vocabulary dựa trên context cho trước.
        """
        distribution = {}
        for word in self.vocab:
            distribution[word] = self.probability(context, word)
        return distribution
    def generate_text(self, max_length=20, prompt=None):
        """
        Sinh câu ngẫu nhiên từ mô hình dựa trên phân phối xác suất.
        :param max_length: Số từ tối đa cần sinh
        :param prompt: Chuỗi từ gợi ý ban đầu (list of str)
        """
        if prompt is None:
            if self.n > 1:
                current_context = tuple(['<s>'] * (self.n - 1))
            else:
                current_context = ()
            generated = []
        else:
            generated = list(prompt)
            if self.n > 1:
                # Lấy n-1 từ cuối của prompt làm context ban đầu
                padded_prompt = ['<s>'] * max(0, (self.n - 1) - len(prompt)) + prompt
                current_context = tuple(padded_prompt[-(self.n - 1):])
            else:
                current_context = ()

        for _ in range(max_length):
            # Lấy phân phối xác suất từ tiếp theo
            dist = self.next_word_distribution(current_context)
            
            # Loại bỏ các token đặc biệt không muốn xuất hiện trực tiếp trong câu sinh ra
            words = list(dist.keys())
            probs = list(dist.values())
            
            # Tối ưu: Nếu dùng MLE, lọc bỏ các từ có xác suất = 0
            valid_words_probs = [(w, p) for w, p in zip(words, probs) if p > 0 and w != '<s>']
            
            if not valid_words_probs:
                break
                
            words, probs = zip(*valid_words_probs)
            
            # Chuẩn hóa lại tổng xác suất bằng 1
            total_p = sum(probs)
            probs = [p / total_p for p in probs]

            # Sampling từ tiếp theo dựa trên trọng số xác suất (Weighted Random Choice)
            next_word = random.choices(words, weights=probs, k=1)[0]

            # Nếu gặp token kết thúc câu </s> thì dừng sinh
            if next_word == '</s>':
                break

            generated.append(next_word)

            # Cập nhật context cho bước tiếp theo
            if self.n > 1:
                current_context = tuple((list(current_context) + [next_word])[1:])

        return " ".join(generated)
    def score_continuation(model, context_tokens, candidate_tokens):
    
        full_tokens = context_tokens + candidate_tokens
    
        # Lấy danh sách các n-gram của toàn bộ chuỗi (context + candidate)
        ngrams = model._get_ngrams(full_tokens)
    
        # Chỉ tính log prob cho các n-gram thuộc phần candidate (tính từ vị trí sau context)
        # Số n-gram của context là len(context_tokens)
        candidate_ngrams = ngrams[len(context_tokens):]
    
        # Tính tổng log probability của các n-gram trong candidate
        total_log_prob = 0.0
        for ngram in candidate_ngrams:
            context = ngram[:-1]
        word = ngram[-1]
        p = model.probability(context, word)
        
        if p == 0.0:
            return float('-inf')
        total_log_prob += math.log(p)
        
        return total_log_prob

def rank_candidates(model, context_str, candidates_list):
    """
    Xếp hạng các câu ứng viên theo log probability giảm dần.
    """
    context_tokens = model._get_ngrams(context_str.lower().split()) # Tách từ đơn giản
    # Lấy các token từ context
    context_tokens = [w for sentence in [context_str.lower().split()] for w in sentence]
    
    ranked_results = []
    for cand in candidates_list:
        cand_tokens = cand.lower().split()
        score = score_continuation(model, context_tokens, cand_tokens)
        ranked_results.append((cand, score))
        
    # Sắp xếp giảm dần theo điểm log prob
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results

if __name__ == "__main__":
    # Corpus thử nghiệm
    corpus = [
        ["the", "cat", "eats", "fish"],
        ["the", "cat", "likes", "fish"],
        ["the", "dog", "eats", "meat"]
    ]

    # Test Bigram Model không smoothing
    model = NGramLanguageModel(n=2, smoothing=False)
    model.fit(corpus)

    print("Vocabulary Size:", len(model.vocab))
    print("P(cat | the):", model.probability(["the"], "cat"))
    print("P(dog | the):", model.probability(["the"], "dog"))
    print("Log Prob sentence 'the cat eats fish':", model.sentence_log_probability(["the", "cat", "eats", "fish"]))