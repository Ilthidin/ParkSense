import streamlit as st

st.set_page_config(
    page_title="Abhinaya-Home",
    page_icon="🏚",
)
st.sidebar.success("Select a page above.")

st.markdown('<div class="logo img"><img src="https://scontent.fsrg13-1.fna.fbcdn.net/v/t39.30808-6/245702209_435359388018255_7220637423361619769_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=5f2048&_nc_eui2=AeHZyVJKTwjKDjUwvZ4Kt-v0-7PYiA2Z2XD7s9iIDZnZcEQykDtrrrDPj8WQsvxU8blf1x7IrvHBNmcM6JZb6aeF&_nc_ohc=h7iSb7dWeMAAX-jVp3B&_nc_ht=scontent.fsrg13-1.fna&oh=00_AfCSKZLghsFPQ_jGQZ4-UTO1dTzBhbt_hrYGnTmIzcGp4A&oe=6559C643"/></div>', unsafe_allow_html=True)
# Add CSS to display the logo
st.markdown(
    """
    <style>
    .logo {
        display: flex;
        align-text: center;
        margin-left: 7%;
    }
    .logo img {
        width: 500px; /* Adjust the width of the logo */
        align-text: center;
        margin-left: 7%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div style="text-align: center; font-size: 30px; font-weight: bold; padding-top: 20px">\
            Project Akhir Kelompok 1 Abhinaya Kelas Athena</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 25px; padding-bottom: 20px">Deteksi Hunian Parkir Menggunakan Computer Vision</div>', unsafe_allow_html=True)
st.image("https://ik.imagekit.io/carro/jualo-public/blog/wp-content/uploads/2022/06/parkir-header.jpg", caption='Deteksi Hunian Parkir')
st.markdown('<div style="text-align: justify; font-size: 20px">Deteksi hunian parkir menggunakan computer vision adalah inovasi untuk mengelola ruang parkir\
             dengan efisien. Teknologi ini mengidentifikasi ruang parkir kosong atau terisi dengan akurasi tinggi, memungkinkan pengoptimalan\
             penggunaan ruang parkir secara real-time. Sistem ini mengurangi waktu pencarian tempat parkir, mengurangi kemacetan, serta memberikan informasi\
             tepat kepada pengguna untuk menemukan tempat parkir kosong. Dampaknya meliputi peningkatan pengalaman pengguna, pengurangan emisi gas\
             buang, dan efisiensi manajemen lahan parkir untuk pemiliknya. Dengan terus berkembangnya teknologi computer vision, deteksi hunian parkir akan\
             terus meningkatkan efisiensi pengelolaan parkir, bahkan berpotensi diintegrasikan dengan sistem transportasi pintar di masa depan.</div>', unsafe_allow_html=True)