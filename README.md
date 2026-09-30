# LAB 02: Language Models - N-gram, Smoothing & Perplexity

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Học kỳ:** Học kỳ 1 - 2026  
**Sinh viên:** [Họ và tên sinh viên]  
**Mã sinh viên:** [Mã sinh viên]  

---

## 📄 Tổng quan dự án

Dự án này triển khai một hệ thống **N-gram Language Model** từ đầu (scratch) bằng Python và NumPy, khảo sát trên tập dữ liệu C4 (10K - 30K văn bản)[cite: 5]. Dự án tập trung giải quyết các bài toán cốt lõi trong xây dựng mô hình ngôn ngữ truyền thống:
- Ước lượng xác suất MLE vs. Laplace (Add-one) Smoothing[cite: 5].
- Đánh giá chỉ số Perplexity (PPL) trên các tập Train / Validation / Test[cite: 5].
- Các ứng dụng: Sinh văn bản (Text Generation) và Xếp hạng câu (Sentence Ranking)[cite: 5].
- Phân tích lỗi (Error Analysis) và ảnh hưởng của chiều dài ngữ cảnh (Context Length)[cite: 5].

---

## 📁 Cấu trúc thư mục Lab

```text
lab02/
├── README.md           # Báo cáo tổng quan dự án
├── calculations.md     # Bài tập tính toán bằng tay (Mục 7, 11, 18)
├── prediction.md       # Các dự đoán trước khi chạy thực nghiệm (Mục 12)
├── ngram_lm.py         # Cài đặt lớp NGramLanguageModel chính
├── experiments.ipynb   # Jupyter Notebook chạy thực nghiệm và trực quan hóa
├── results.csv         # Bảng tổng hợp kết quả Perplexity và Ranking
├── error_analysis.md   # Phân tích 2 trường hợp dự đoán đúng & 2 trường hợp sai
└── reflection.md       # Trả lời 7 câu hỏi tổng kết và định hướng Neural LM