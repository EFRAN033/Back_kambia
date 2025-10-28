# create_tables.py
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Importa tus modelos y la Base desde main.py
# Asegúrate de que los modelos SiteSettings y GalleryItem estén definidos
# y asociados a la variable 'Base' en tu main.py
from main import Base 

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("La variable de entorno DATABASE_URL no está configurada...")

engine = create_engine(DATABASE_URL)

print("Intentando crear las tablas que faltan (SiteSettings, GalleryItem)...")

# Esta línea creará las tablas en la base de datos
Base.metadata.create_all(bind=engine)

print("¡Tablas creadas con éxito! Ahora puedes iniciar la aplicación.")