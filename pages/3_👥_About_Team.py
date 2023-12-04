import streamlit as st

st.set_page_config(
    page_title="Abhinaya-About Team",
    page_icon="👥",
)
st.title("About Team")

# Gambar tim beserta universitas
team_data = {
    "Muhammad Syihab Habibi": {
        "university": "Universitas Sultan Agung",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/syihab.png?raw=true"
    },
    "Fiqri Ramadhan": {
        "university": "IKIP Siliwangi",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/fiqri.png?raw=true"
    },
    "Siti Choiriyah": {
        "university": "Universitas Stikubank",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/siti.png?raw=true"
    },
    "Echa Oktamiani Maulana": {
        "university": "Universitas Islam 45",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/echa.png?raw=true"
    },
    "Chalda Bhakti Jelika": {
        "university": "Universitas Ahmad Dahlan",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/chalda.png?raw=true"
    },
    "Imroatu Solicah": {
        "university": "Universitas Stikubank",
        "image_url": "https://github.com/hallosayaimroatubelajargithub/kelompok-abhinaya-ofa/blob/main/images/imroatu.png?raw=true"
    },
}

# Menampilkan gambar untuk setiap anggota tim secara horizontal beserta universitasnya
row = '<div style="display:flex; flex-wrap:wrap;">'
for member, data in team_data.items():
    university = data["university"]
    image_url = data["image_url"]
    row += f'<div style="margin:10px;"><img src="{image_url}" style="width:200px;"><p style="text-align:center;">{member}<br>{university}</p></div>'
row += '</div>'

st.markdown(row, unsafe_allow_html=True)

