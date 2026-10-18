# 🔥 Coal Algo-Trader v1.0

**Sistem Rekomendasi Saham Batu Bara BEI Berbasis Algoritma Dual-Engine**

> ⚡ **Swing Trading Mode** — Capital Gain Hunter (3 hari - 4 minggu)  
> 💰 **Dividend Investing Mode** — Yield Accumulator (Jangka Panjang)

---

## 📋 Daftar Isi

1. [Tentang Proyek](#-tentang-proyek)
2. [Fitur Utama](#-fitur-utama)
3. [Arsitektur Sistem](#-arsitektur-sistem)
4. [Teknologi](#-teknologi)
5. [Instalasi](#-instalasi)
6. [Cara Menggunakan](#-cara-menggunakan)
7. [Konfigurasi](#-konfigurasi)
8. [Struktur File](#-struktur-file)
9. [Deployment](#-deployment)
10. [Contoh Output](#-contoh-output)
11. [Troubleshooting](#-troubleshooting)
12. [Roadmap](#-roadmap)
13. [Disclaimer](#-disclaimer)

---

##  Tentang Proyek

**Coal Algo-Trader v1.0** adalah sistem rekomendasi saham batu bara di Bursa Efek Indonesia (BEI) yang menggunakan pendekatan **Dual-Engine Architecture** untuk menghasilkan sinyal trading dan investasi yang berbeda sesuai profil masing-masing:

| Engine | Profil | Horizon | Fokus |
|--------|--------|---------|-------|
| **Swing Mode** | Capital Gain Hunter | 3 hari - 4 minggu | Momentum, Technical, Volume |
| **Dividend Mode** | Yield Accumulator | Tahunan | Fundamental, Dividen, Fear/Panic |

Sistem ini **otomatis** — berjalan setiap hari kerja pukul 16:30 WIB via **GitHub Actions**, menghasilkan file `coal_data.json` yang di-render oleh frontend **Detik.com-style** yang di-host di **GitHub Pages**.

### 📈 Ticker yang Dianalisis
