import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import time

st.set_page_config(page_title="Tumor Otak.AI")
# Memuat model yang telah dilatih
model = load_model('keras_model.h5')

def prediksi_gambar(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 225.

    prediksi = model.predict(img_tensor)
    kelas_label = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    indeks_kelas = np.argmax(prediksi[0])
    label_kelas = kelas_label[indeks_kelas]
    skor_kepercayaan = float(prediksi[0][indeks_kelas])

    hasil = {
        'label_kelas': label_kelas,
        'skor_kepercayaan': skor_kepercayaan,
        'skor_kesalahan' : 1-skor_kepercayaan
    }

    return hasil

# Aplikasi Streamlit

# Navigasi
halaman_terpilih = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Halaman Prediksi", "Visualisasi Model"],format_func=lambda x: x)

if halaman_terpilih == "Beranda":
    # Tampilkan Halaman Beranda

    st.header("Selamat Datang di Aplikasi Prediksi Tumor Otak!", divider='rainbow')
    st.write(
        "Aplikasi ini memungkinkan Anda untuk mengunggah gambar Otak Anda"
        " dan mendapatkan hasil prediksi apakah Otak Anda Normal atau terdeteksi Tumor Otak."
    )
    st.write("Silahkan Pilih 'Halaman Prediksi' untuk memulai prediksi.")

elif halaman_terpilih == "Halaman Prediksi":
    # Tampilkan Halaman Prediksi
    st.title("Unggah Gambar")
    st.markdown("---")

    # Unggah Gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silahkan Pilih Gambar ", type=["jpg", "jpeg", "png"])

    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang Diunggah", use_column_width=True)

        # Lakukan prediksi saat tombol ditekan
        if st.button("Prediksi"):
            # Simpan berkas gambar yang diunggah ke lokasi sementara
            with open("temp_image.jpg", "wb") as f :
                f.write(berkas_gambar.getbuffer())

            # Lakukan Prediksi Pada Berkas yang disimpan
            hasil_prediksi = prediksi_gambar("temp_image.jpg")

            # Tampilkan Hasil Prediksi
            st.write(f"Prediksi: {hasil_prediksi['label_kelas']}")
            st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")
            st.write(f"Skor Kesalahan: {hasil_prediksi['skor_kesalahan']:.2%}")

            if hasil_prediksi['label_kelas'] == 'Normal' :
                st.write("Selamat! Berdasarkan prediksi kami, Esophagus Anda tampaknya dalam keadaan normal. Namun, Ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami. Jika Anda memiliki kekhawatiran kesehatan atau pertanyaan lebih lanjut, sangat disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan yang lebih mendalam.")
            
            elif hasil_prediksi['label_kelas'] == 'glioma' :
                st.write("Hasil prediksi menunjukkan kemungkunan terdeteksi adanya Tumor pada Otak Anda. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan anda.")
            
            elif hasil_prediksi['label_kelas'] == 'meningioma' :
                st.write("Hasil prediksi menunjukkan kemungkunan terdeteksi adanya Tumor pada Otak Anda. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan anda.")
            
            elif hasil_prediksi['label_kelas'] == 'pituitary' :
                st.write("Hasil prediksi menunjukkan kemungkunan terdeteksi adanya Tumor pada Otak Anda. Namun, perlu diingat bahwa ini hanya hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan anda.")        

else:
    # Aplikasi Streamlit
    st.title("Visualisasi Model Ai")
    st.markdown("---")

    def display_image_table(image_path1, title1, caption1, image_path2, title2, caption2):
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2) if image_path2 else None

        col1, col2 = st.columns(2)

        with col1:
            col1.markdown(f'<h2 style="text-align:center;">{title1}</h2>', unsafe_allow_html=True)
            col1.markdown(
                f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
            )
            col1.image(image1, use_column_width=True)
            col1.markdown(f'<p stye="text-align:left;">{caption1}</p>', unsafe_allow_html=True)
        
        with col2:
            if image2:
                col2.markdown(f'<h2 style="text-align:center;">{title2}</h2>', unsafe_allow_html=True)
            
                col2.markdown(
                    f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
                )
                col2.image(image2, use_column_width=True)
                col2.markdown(f'<p stye="text-align:left;">{caption2}</p>', unsafe_allow_html=True)

    images_info = [
        {'path': 'accuracy_perclass.png','title': 'Accuracy Class', 'caption':'Tabel ini menunjukkan tingkat akurasi model klasifikasi untuk masing-masing kelas tumor, dengan akurasi tertinggi pada kelas no_tumor dan pituitary_tumor (0.94) serta akurasi terendah pada kelas meningioma_tumor (0.87).'},
        {'path': 'accuracy_perepoch.png','title': 'Accuracy Epoch', 'caption':'Grafik ini menunjukkan kinerja model dalam hal akurasi selama pelatihan, bahwa akurasi pelatihan (acc) cepat mencapai hampir 100%, sementara akurasi pengujian (test acc) stabil di sekitar 85% setelah beberapa epoch, mengindikasikan potensi overfitting.'},
        {'path': 'confusion_matrix.png','title': 'Confusion Matrix', 'caption':'Matriks menunjukkan confusion matrix yang menilai kinerja model klasifikasi tumor otak, di mana prediksi paling akurat terjadi pada kelas glioma_tumor dan pituitary_tumor, sementara kelas meningioma_tumor dan no_tumor menunjukkan beberapa kesalahan prediksi.'},
        {'path': 'loss_perepoch.png','title': 'Loss Epoch', 'caption':'Grafik ini menunjukkan bahwa loss pada data pelatihan (loss) cepat menurun mendekati nol, sedangkan loss pada data pengujian (test loss) tetap relatif tinggi dan stabil, mengindikasikan bahwa model mengalami overfitting.'},
    ]
    for i in range(0, len(images_info), 2):
        if i + 1 < len(images_info):
            display_image_table(
                images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'],
                images_info[i + 1]['path'], images_info[i + 1]['title'], images_info[i + 1]['caption']
            )
        else:
            display_image_table(images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'], '', '', '')
