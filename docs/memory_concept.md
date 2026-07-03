# Konsep Memori Percakapan pada RAG Assistant

Dokumen ini menjelaskan bagaimana fitur memori (Conversation Memory) bekerja di balik layar dalam aplikasi Personal RAG Assistant ini.

Sistem memori ini terdiri dari dua siklus utama: **Store Memory** (Menyimpan Memori) dan **Recall Memory** (Memanggil Memori). Pendekatan ini memungkinkan LLM untuk menanggapi pertanyaan lanjutan (*follow-up questions*) seolah-olah ia sedang berdialog dengan manusia.

---

## 1. Store Memory (Menyimpan Memori)

Proses penyimpanan memori terjadi di *frontend* (Streamlit) segera setelah ada interaksi dengan pengguna.

**Cara kerjanya:**
1. Setiap kali pengguna mengirim pertanyaan, teks tersebut langsung disimpan ke dalam `st.session_state.messages` dengan format JSON:
   `{"role": "user", "content": "Apa itu Machine Learning?"}`
2. Setelah sistem selesai memproses dan menjawab, respons dari AI juga ikut disimpan:
   `{"role": "assistant", "answer": "Machine Learning adalah..."}`
3. Data sesi sementara ini kemudian dikunci (persisten) dalam penyimpanan cloud (Supabase/JSON lokal tergantung konfigurasi) agar tidak hilang saat pengguna me-*refresh* halaman.

*Kode Relevan: `app.py` (bagian pengelolaan `st.session_state.messages`)*

---

## 2. Recall Memory (Memanggil Memori)

Fase ini adalah jantung dari integrasi LangChain. Ada dua proses kritis di mana sistem "mengingat" percakapan sebelumnya untuk menghasilkan konteks yang tepat:

### A. Recall saat Retrieval (Query Reformulation)
Masalah terbesar di RAG tradisional adalah ketika pengguna bertanya *"Jelaskan lebih detail"*, sistem akan mencari dokumen yang mengandung kata "jelaskan lebih detail" ke dalam *Vector Database* (ChromaDB), yang tentunya akan gagal.

*Recall Memory* menyelesaikan ini dengan:
1. `rag_pipeline.py` mengambil 6 interaksi terakhir dari memori (setara 3 tanya-jawab).
2. Sistem menyisipkan riwayat tersebut beserta pertanyaan baru ke dalam `reformulation_prompt`.
3. LLM kemudian menganalisis konteksnya dan mengubah pertanyaan asli menjadi **Standalone Query** yang spesifik.
   *(Misal: "Jelaskan lebih detail" → "Jelaskan lebih detail tentang tipe-tipe Machine Learning berdasarkan dokumen")*
4. *Vector Database* kemudian menggunakan query baru ini untuk menarik dokumen yang relevan.

### B. Recall saat Generasi Jawaban (Answer Generation)
Setelah dokumen yang relevan ditemukan, AI juga harus membaca ulang riwayat percakapan agar jawabannya selaras dengan percakapan sebelumnya.

1. `generator.py` memuat ulang daftar percakapan sebelumnya ke dalam *System/User Prompt* (via `ChatPromptTemplate`).
2. *System Prompt* memiliki instruksi (RULE 6) yang memaksa LLM untuk selalu mengecek konteks `Conversation History` apabila menjumpai kata ganti seperti "itu", "dia", atau "hal tersebut".
3. Hasil akhirnya, LLM bisa memberikan jawaban yang akurat, berbobot, dan berkelanjutan (*continuous*).

---

## Kesimpulan
Dengan memisahkan antara **Store Memory** (di sisi UI) dan **Recall Memory** (di sisi RAG Logic dan Retrieval), aplikasi ini dapat mempertahankan kecepatan tanpa membebani ukuran *prompt*, serta sepenuhnya mencegah model salah tafsir pada pertanyaan bersambung.
