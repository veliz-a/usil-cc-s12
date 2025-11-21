#%%
import streamlit as st
from supabase import create_client
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import base64
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')
#%%
# Si sólo usas anon key para cliente, puedes usar SUPABASE_ANON_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error('Define SUPABASE_URL y SUPABASE_KEY en tu .env antes de ejecutar la app')
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title='Digits + Supabase', layout='centered')

# Entrenar modelo de sklearn (rápido usando el dataset digits)
@st.cache_resource
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model()

# Preprocesar imagen para que se parezca a los 8x8 de sklearn
def preprocess_to_8x8(img: Image.Image):
    # img: PIL Image
    # convert to grayscale
    img = ImageOps.grayscale(img)
    # resize to 8x8
    img = img.resize((8,8), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=float)
    # invert colors si el fondo es blanco y dígito negro -> queremos similar a dataset
    # normalize a 0-16 como en load_digits
    # Primero, scale 0-255 -> 0-16
    arr = 16.0 * (255.0 - arr) / 255.0
    flat = arr.flatten()
    return flat

# Categorizar IMC
def categorize_bmi(bmi: float):
    if bmi < 18.5:
        return 'Bajo peso'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Sobrepeso'
    else:
        return 'Obesidad'

# Autenticación simple: login / register
st.title('App: Reconocimiento de dígitos + IMC (Supabase)')

menu = st.sidebar.selectbox('Menú', ['Entrar / Registrar', 'App'])

if menu == 'Entrar / Registrar':
    st.header('Autenticación')
    action = st.radio('Acción', ['Entrar', 'Registrar'])
    email = st.text_input('Email')
    password = st.text_input('Contraseña', type='password')

    if action == 'Registrar':
        if st.button('Crear cuenta'):
            if not email or not password:
                st.warning('Ingresa email y contraseña')
            else:
                try:
                    res = supabase.auth.sign_up({ 'email': email, 'password': password })
                    # res contiene info o error
                    st.success('Cuenta creada. Revisa tu correo para confirmar (si está habilitado).')
                except Exception as e:
                    st.error(f'Error al registrar: {e}')

    else:  # Entrar
        if st.button('Entrar'):
            if not email or not password:
                st.warning('Ingresa email y contraseña')
            else:
                try:
                    res = supabase.auth.sign_in_with_password({ 'email': email, 'password': password })
                    # la API reciente usa sign_in_with_password
                    session = res
                    if session and session.session.access_token:
                        st.success('Autenticado')
                        st.session_state['user'] = session['user'] if 'user' in session else {'email': email}
                    else:
                        st.error('No se pudo autenticar. Revisa credenciales')
                except Exception as e:
                    st.error(f'Error al autenticar: {e}')

    # Botón para cerrar sesión
    if st.button('Cerrar sesión (local)'):
        try:
            supabase.auth.sign_out()
        except:
            pass
        st.session_state.pop('user', None)
        st.success('Sesión cerrada')

# Página principal de la app
if menu == 'App':
    user = st.session_state.get('user')
    if not user:
        st.info('Por favor entra o registra una cuenta en el menú "Entrar / Registrar"')
    else:
        st.write('Conectado como:', user.get('email'))
        with st.form('form_upload'):
            st.subheader('Sube una imagen con un dígito (o dibuja y sube)')
            uploaded = st.file_uploader('Imagen (PNG/JPG)', type=['png','jpg','jpeg'])
            weight = st.number_input('Peso (kg)', min_value=0.0, max_value=500.0, value=70.0, step=0.1)
            height_cm = st.number_input('Altura (cm)', min_value=50.0, max_value=300.0, value=170.0, step=0.1)
            submitted = st.form_submit_button('Predecir y guardar')

        if submitted:
            if uploaded is None:
                st.warning('Sube una imagen para predecir')
            else:
                try:
                    image = Image.open(uploaded).convert('RGB')
                    st.image(image, caption='Imagen subida', use_container_width=False)

                    features = preprocess_to_8x8(image)
                    pred = model.predict([features])[0]
                    st.success(f'Predicción: {pred}')

                    # Calcular IMC
                    height_m = height_cm / 100.0
                    bmi = weight / (height_m**2) if height_m > 0 else None
                    bmi_cat = categorize_bmi(bmi) if bmi else 'N/A'
                    st.write(f'IMC: {bmi:.2f} — {bmi_cat}')

                    # Guardar registro en Supabase
                    user_email = user.get('email')
                    # Generar nombre de archivo único
                    from datetime import datetime, timezone
                    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    filename = f"{user_email.replace('@','_at_')}_{ts}.png"

                    # Subir imagen al bucket llamado 'digits-images' (asegúrate de crearlo en Supabase)
                    buffered = BytesIO()
                    image.save(buffered, format='PNG')
                    buffered.seek(0)

                    try:
                        file_bytes = buffered.getvalue()
                        # upload
                        res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, file_bytes, {'content-type':'image/png'})
                        # obtener public URL
                        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
                    except Exception as e:
                        st.error(f'Error subiendo imagen al storage: {e}')
                        public_url = None

                    # Insertar en la tabla users_measurements
                    record = {
                        'user_id': user.get('id', user_email),
                        'email': user_email,
                        'weight_kg': float(weight),
                        'height_m': float(height_m),
                        'bmi': float(bmi),
                        'bmi_category': bmi_cat,
                        'image_path': public_url
                    }
                    try:
                        insert = supabase.table('users_measurements').insert(record).execute()
                        st.success('Datos guardados en Supabase')
                    except Exception as e:
                        st.error(f'Error guardando en Postgres: {e}')

                except Exception as e:
                    st.error(f'Error procesando la imagen: {e}')

        st.markdown('---')
        st.subheader('Tus datos recientes')
        try:
            resp = supabase.table('users_measurements').select('*').eq('email', user.get('email')).order('created_at', desc=True).limit(5).execute()
            rows = resp.data if hasattr(resp, 'data') else resp
            if rows:
                for r in rows:
                    st.write(f"{r.get('created_at')}: IMC={r.get('bmi'):.2f} ({r.get('bmi_category')}), imagen: {r.get('image_path')}")
            else:
                st.info('No hay registros todavía')
        except Exception as e:
            st.error(f'No se pudieron recuperar registros: {e}')