import os
import json
import openai
from datetime import datetime

# Inisialisasi OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load titles dari JSON
with open('titles.json', 'r', encoding='utf-8') as f:
    titles = json.load(f)

# Master Prompt Template
SYSTEM_PROMPT = """
Kamu adalah seorang narator dokumenter investigasi profesional dengan 
gaya seperti Vice News, Lemmings, atau Johnny Harris. Nada bicaramu 
serius, analitis, penuh penekanan, dan menggugah rasa ingin tahu. 
Kamu tidak menggunakan bahasa gaul, tidak menggunakan filler, dan 
tidak basa-basi.

TUGAS:
Buat skrip storytelling YouTube untuk video berdurasi 20-25 menit 
(sekitar 3000-3500 kata) dengan topik berikut:

ATURAN PENULISAN:
- Gunakan kalimat efektif, hindari repetisi
- Setiap klaim HARUS disertai data, contoh, atau logika
- Gunakan metafora yang kuat (rantai, penjara, mesin, racun)
- Sertakan tanda baca yang tepat untuk jeda napas VO
- Nada: serius seperti membongkar kasus kriminal
- BAHASA: Indonesia formal namun mengalir
- LARANGAN: jangan gunakan kata "guys", "banget", "ya guys"
- LARANGAN: jangan bertele-tele tanpa esensi

STRUKTUR WAJIB:
1. HOOK (00:00 - 02:30)
2. BAB I - TITIK NOL SEJARAH (02:30 - 07:00)
3. BAB II - MESIN DI BALIK LAYAR (07:00 - 11:30)
4. BAB III - DAMPAK NYATA (11:30 - 16:00)
5. BAB IV - PSIKOLOGI KORBAN (16:00 - 20:00)
6. BAB V - ANALISIS SEBAB-AKIBAT (20:00 - 24:00)
7. KESIMPULAN & REFLEKSI (24:00 - 26:00)

OUTPUT: Berikan skrip lengkap dengan penanda [BAGIAN X: NAMA BAB]
"""

def generate_script(title_data):
    """Generate script untuk satu judul"""
    
    user_prompt = f"""
[JUDUL VIDEO]: {title_data['title']}
[TEMA BESAR]: {title_data['theme']}
[SUBTITLE]: {title_data.get('subtitle', '')}

Buat skrip lengkap sesuai struktur yang diminta.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Atau "gpt-3.5-turbo" untuk lebih murah
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.7
        )
        
        script_content = response.choices[0].message.content
        
        # Simpan ke file
        filename = f"output/script_{title_data['id']:02d}_{sanitize_filename(title_data['title'])}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"JUDUL: {title_data['title']}\n")
            f.write(f"TEMA: {title_data['theme']}\n")
            f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(script_content)
        
        print(f"✅ Script berhasil dibuat: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error untuk judul '{title_data['title']}': {str(e)}")
        return False

def sanitize_filename(title):
    """Bersihkan judul untuk jadi nama file"""
    return title.replace("!", "").replace("=", "-").replace(" ", "_")[:50]

def main():
    print("🎬 Memulai generasi script YouTube...\n")
    
    # Buat folder output jika belum ada
    os.makedirs('output', exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for title_data in titles:
        print(f"📝 Memproses: {title_data['title']}")
        
        if generate_script(title_data):
            success_count += 1
        else:
            failed_count += 1
        
        # Delay untuk rate limiting
        import time
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f" SELESAI!")
    print(f"✅ Berhasil: {success_count} script")
    print(f"❌ Gagal: {failed_count} script")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
