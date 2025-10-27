from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Date, Text, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Literal, Set
from passlib.context import CryptContext
from sqlalchemy import Table
from sqlalchemy import func
from fastapi import WebSocket, WebSocketDisconnect
# Importaciones para JWT
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
import aiofiles
import uuid
import json
from sqlalchemy import or_, and_
import mercadopago
from typing import List, Dict
import asyncio
from broadcaster import Broadcast
from sqlalchemy.orm import joinedload
#Importaciones para la validacion de dni con PeruDevs
import requests
import re

hero_section_data = {
    # Textos generales (como los tenías)
    "titleLine1": "Intercambia fácil, seguro",
    "titleLine2": "y sin comisiones",
    "badgeText": "Bienvenido a KambiaPe",
    "paragraphText": "Publica lo que ya no usas y encuentra lo que necesitas en tu comunidad.",
    "button1Text": "Publicar objeto",
    "button2Text": "Buzón",

    # Lista de tarjetas con valores por defecto
    "cards": [
        {
            "id": 1,
            "alt": "Cámara",
            "badge": "Destacado",
            # Usa rutas relativas si las imágenes están en /uploaded_images/ o None si no hay default
            "imageUrl": "/src/assets/imagenes/gif/Animacion_Mesa de trabajo 1-01.png" # O None
        },
        {
            "id": 2,
            "alt": "Mochila",
            "badge": "Outdoor",
            "imageUrl": "/src/assets/imagenes/gif/Animacion_Mesa de trabajo 1-02.png" # O None
        },
        {
            "id": 3,
            "alt": "Teclado",
            "badge": "Gaming",
            "imageUrl": "/src/assets/imagenes/gif/Animacion_Mesa de trabajo 1-03.png" # O None
        }
    ]
}

# --- Datos para la página "Nosotros" ---
about_us_data = {
  "hero": {
    "badge": "Perú • Comunidad • Innovación",
    "title": "Ideas en acción.",
    "paragraph": "KambiaPe conecta personas para transformar barrios, escuelas y emprendimientos. Simple, local, real.",
    "btn1": "Conoce la historia",
    "btn2": "Conócenos!"
  },
  "heroCards": [
    { "id": 1, "alt": "Mentoría STEAM en colegio público", "caption": "Educación", "title": "Escuelas que inspiran", "imageUrl": "/uploads/placeholder1.jpg" },
    { "id": 2, "alt": "Reciclaje y compostaje comunitario", "caption": "Sostenibilidad", "title": "Rutas verdes", "imageUrl": "/uploads/placeholder2.jpg" },
    { "id": 3, "alt": "Emprendimiento local y comercio digital", "caption": "Emprendimiento", "title": "Impulso emprendedor", "imageUrl": "/uploads/placeholder3.jpg" }
  ],
  "about": {
    "title": "Trueque con propósito.",
    "paragraph": "Creamos puentes entre personas y barrios para que el intercambio vuelva a ser cotidiano, justo y medible.",
    "quote": "Intercambiar es confiar y co-crear."
  },
  "tabs": [
    { "id": 1, "type": "Qué es", "paragraph": "Una red donde productos, servicios y conocimientos se intercambian con reglas claras, reputación visible y mediación. Tecnología al servicio de la confianza local.", "quote": "Lo digital acerca, la comunidad decide.", "list": "", "footer": "", "alt": "Personas colaborando", "caption": "Colaboración", "imageUrl": "/uploads/placeholder4.jpg" },
    { "id": 2, "type": "Misión", "paragraph": "Facilitar el intercambio justo entre personas mediante una plataforma confiable, segura y usable, con foco en impacto local.", "quote": "", "list": "Reglas claras y reputación pública.\nInclusión económica en barrios y escuelas.\nMenos desperdicio a través de la reutilización.", "footer": "", "alt": "Huertos urbanos", "caption": "Reutilización", "imageUrl": "/uploads/placeholder5.jpg" },
    { "id": 3, "type": "Visión", "paragraph": "Liderar el intercambio solidario en Perú con una economía colaborativa y sostenible, conectando comunidades a nivel nacional a través de alianzas locales y tecnología centrada en las personas.", "quote": "", "list": "", "footer": "Alianzas con municipios y escuelas, embajadores locales y un motor de matching...", "alt": "Robótica y formación", "caption": "Formación", "imageUrl": "/uploads/placeholder6.jpg" }
  ],
  "community": {
    "title": "El corazón de KambiaPe es su gente.",
    "paragraph": "Únete a nuestra comunidad en WhatsApp para conectar, compartir ideas y ser el primero en enterarte de las novedades.",
    "btnText": "Unirme a la Comunidad",
    "link": "https://chat.whatsapp.com/Kekd3xJZtet8J6TLhTGFDQ?mode=ems_wa_t"
  },
  "social": {
    "insta": "https://www.instagram.com/kambia_pe?igsh=MWg2aWR3YnhnNW1qdw==",
    "tiktok": "https://tiktok.com/@kambiape",
    "facebook": "https://www.facebook.com/share/1A62pnpV8K/"
  }
}

load_dotenv()

# 1. PRIMERO, defines la variable leyendo el archivo .env
DATABASE_URL = os.getenv("DATABASE_URL")

# 2. LUEGO, la usas para verificar si existe
if not DATABASE_URL:
    raise ValueError("La variable de entorno DATABASE_URL no está configurada...")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
broadcast = Broadcast(REDIS_URL)

# Carga las variables del API de Perudevs para usarlas globalmente
PERUDEVS_API_KEY = os.getenv("PERUDEVS_DNI_KEY")
PERUDEVS_DNI_URL = os.getenv("PERUDEVS_DNI_URL")

# Carga el Access Token de Mercado Pago
MERCADOPAGO_ACCESS_TOKEN = os.getenv("MERCADOPAGO_ACCESS_TOKEN")
if not MERCADOPAGO_ACCESS_TOKEN:
    raise ValueError("La variable de entorno MERCADOPAGO_ACCESS_TOKEN no está configurada.")

# Inicializa el SDK de Mercado Pago
sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)

# Ahora creamos las instancias de la base de datos
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ... el resto de tu código

user_interests_table = Table('user_interests', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('category_id', Integer, ForeignKey('categories.id', ondelete="CASCADE"), primary_key=True)
)

product_exchange_interests_table = Table('product_exchange_interests', Base.metadata,
    Column('product_id', Integer, ForeignKey('products.id', ondelete="CASCADE"), primary_key=True),
    Column('category_id', Integer, ForeignKey('categories.id', ondelete="CASCADE"), primary_key=True)
)

user_blocks = Table('user_blocks', Base.metadata,
    Column('blocker_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('blocked_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True)
)

# --- Modelos de la base de datos (SQLAlchemy) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    agreed_terms = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    phone = Column(String(50), nullable=True)
    ubicacion = Column(String(255), nullable=True)
    district_id = Column(String(50), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)
    occupation = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    dni = Column(String(12), nullable=True, unique=True)
    role = Column(String(50), nullable=False, default='user')
    credits = Column(Integer, default=10, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    interests = relationship("Category", secondary=user_interests_table, back_populates="interested_users", lazy="joined")
    profile_picture = Column(String(500), nullable=True)

    ratings_given = relationship("UserRating", foreign_keys='UserRating.rater_id', back_populates="rater", cascade="all, delete-orphan")
    ratings_received = relationship("UserRating", foreign_keys='UserRating.rated_id', back_populates="rated_user", cascade="all, delete-orphan")

    products_owned = relationship("Product", back_populates="owner")
    proposals_made = relationship("Proposal", foreign_keys="[Proposal.proposer_user_id]", back_populates="proposer")
    proposals_received = relationship("Proposal", foreign_keys="[Proposal.owner_of_requested_product_id]", back_populates="owner_of_requested_product")

    blocked_users = relationship("User", 
                                 secondary=user_blocks,
                                 primaryjoin=(id == user_blocks.c.blocker_id),
                                 secondaryjoin=(id == user_blocks.c.blocked_id),
                                 backref="blocked_by_users")
 
class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    interested_users = relationship("User", secondary=user_interests_table, back_populates="interests")


    products_linked = relationship("Product", back_populates="category_obj")
    sought_by_products = relationship("Product", secondary=product_exchange_interests_table, back_populates="exchange_interests")

class UserReport(Base):
    __tablename__ = 'user_reports'
    id = Column(Integer, primary_key=True, index=True)
    reporter_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"), nullable=False)
    reported_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    reporter = relationship("User", foreign_keys=[reporter_id])
    reported = relationship("User", foreign_keys=[reported_id])

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    
    mp_payment_id = Column(String(255), unique=True, nullable=True, index=True)
    mp_preference_id = Column(String(255), nullable=True)
    external_reference = Column(String(255), nullable=True, index=True)
    
    status = Column(String(50), nullable=False, index=True)
    description = Column(String(255), nullable=True)
    currency_id = Column(String(3), nullable=False, default="PEN")
    amount = Column(DECIMAL(10, 2), nullable=False)
    
    payment_method_id = Column(String(100), nullable=True)
    payment_type_id = Column(String(100), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User")


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    current_value_estimate = Column(DECIMAL(10, 2), nullable=True)
    condition = Column(String(50), nullable=True)
    status = Column(String(50), default='available', nullable=False)
    preffered_exchange_items = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    views_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    


    owner = relationship("User", back_populates="products_owned")
    category_obj = relationship("Category", back_populates="products_linked")
    images = relationship("ProductImage", back_populates="product_obj", cascade="all, delete-orphan", order_by="ProductImage.upload_order")
    proposals_offered = relationship("Proposal", foreign_keys="[Proposal.offered_product_id]", back_populates="offered_product", cascade="all, delete-orphan")
    proposals_requested = relationship("Proposal", foreign_keys="[Proposal.requested_product_id]", back_populates="requested_product", cascade="all, delete-orphan")
    exchange_interests = relationship("Category", secondary=product_exchange_interests_table, back_populates="sought_by_products", lazy="joined")



class ProductImage(Base):
    __tablename__ = "product_images"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    image_url = Column(String(500), nullable=False)
    is_thumbnail = Column(Boolean, default=False, nullable=False)
    upload_order = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    product_obj = relationship("Product", back_populates="images")

class Proposal(Base):
    __tablename__ = "proposals"
    id = Column(Integer, primary_key=True, index=True)
    offered_product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    requested_product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    proposer_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner_of_requested_product_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(50), default="pending", nullable=False) # e.g., 'pending', 'accepted', 'rejected', 'canceled'
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    deleted_by_proposer = Column(Boolean, default=False)
    deleted_by_receiver = Column(Boolean, default=False)

    # Relaciones para acceder a los objetos completos
    offered_product = relationship("Product", foreign_keys=[offered_product_id], back_populates="proposals_offered")
    requested_product = relationship("Product", foreign_keys=[requested_product_id], back_populates="proposals_requested")
    proposer = relationship("User", foreign_keys=[proposer_user_id], back_populates="proposals_made")
    owner_of_requested_product = relationship("User", foreign_keys=[owner_of_requested_product_id], back_populates="proposals_received")
    
    # NUEVA relación para los mensajes asociados a esta propuesta
    messages = relationship("Message", back_populates="proposal", cascade="all, delete-orphan", order_by="Message.timestamp")


# --- INICIO DE CAMBIO 1: Modelo Message ---
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer, ForeignKey("proposals.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=True) # <-- MODIFICADO: Ahora puede ser nulo
    image_url = Column(String(500), nullable=True) # <-- NUEVO: Para guardar la URL de la imagen
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_read = Column(Boolean, default=False)

    proposal = relationship("Proposal", back_populates="messages")
    sender_obj = relationship("User", foreign_keys=[sender_id], lazy="joined")
# --- FIN DE CAMBIO 1 ---
class UserRatingBase(BaseModel):
    score: int = Field(..., gt=0, le=5)
    comment: Optional[str] = None

class UserRatingCreate(UserRatingBase):
    rated_id: int
    proposal_id: int

class AdminCreditUpdate(BaseModel):
    credits: int = Field(..., ge=0) 

class AdminStatusUpdate(BaseModel):
    is_active: bool

class UserRatingResponse(UserRatingBase):
    id: int
    rater_id: int
    rated_id: int
    proposal_id: int
    created_at: datetime

    class Config:
        from_attributes = True 

class CreditPurchaseRequest(BaseModel):
    quantity: int = Field(..., gt=0) 
    unit_price: float = Field(..., gt=0.0)
    title: str = "Compra de Créditos para KambiaPe"

class HeroCard(BaseModel):
    id: int
    alt: str
    badge: str
    imageUrl: Optional[str] = None    

class UserReportResponse(BaseModel):
    id: int
    reporter_id: int
    reported_id: int
    reason: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class PaymentRequest(BaseModel):
    token: str
    issuer_id: str
    payment_method_id: str
    transaction_amount: float
    installments: int
    payer: dict
    description: str

Base.metadata.create_all(bind=engine, checkfirst=True)

app = FastAPI(
    title="KambiaPe API",
    description="API para la gestión de usuarios y transacciones de KambiaPe.",
    version="0.0.1",
)

from fastapi.middleware.cors import CORSMiddleware

# CONFIGURACIÓN DE CORS
origins = [
    "http://localhost:5173",    # La dirección principal de Vite
    "http://127.0.0.1:5173",
    "https://kambiape.com",
    "http://kambiape.com",  # A veces el navegador usa esta IP, es bueno tenerla
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_dni_data(dni: str, full_name: str) -> dict: # <--- CAMBIO: Añadir el tipo de retorno 'dict'
    """
    Consulta el API de DNI Completo de Perudevs. 
    Verifica que el DNI coincida con el nombre y devuelve los datos biográficos.
    """
    
    if not PERUDEVS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La clave de API de validación de DNI no está configurada en el servidor."
        )
    
    if not re.match(r"^\d{8}$", dni):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El DNI debe ser un valor de 8 dígitos."
        )

    try:
        response = requests.get(
            PERUDEVS_DNI_URL, # Usará la URL de DNI Completo
            params={"document": dni, "key": PERUDEVS_API_KEY},
            timeout=5
        )
        response.raise_for_status()

        data = response.json()

        if not data.get("estado"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=data.get("mensaje", "DNI no encontrado o no activo en el registro nacional.")
            )

        resultado = data.get("resultado", {})
        nombre_api = resultado.get("nombre_completo", "").strip().upper()
        
        nombre_usuario_normalizado = " ".join(full_name.strip().upper().split())
        
        # Lógica de validación: Comparamos el nombre.
        if nombre_api != nombre_usuario_normalizado:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre completo no coincide con el número de DNI proporcionado. Usa tu nombre legal."
            )
        
        # SI LA VALIDACIÓN ES EXITOSA, DEVUELVE LOS DATOS COMPLETOS
        return resultado # <--- CAMBIO CLAVE: Devuelve el diccionario de resultados

    except requests.RequestException as e:
        print(f"Error al consultar API de Perudevs: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fallo al verificar el DNI. Inténtalo más tarde."
        )

class UserCreate(BaseModel):
    full_name: str
    dni: str
    email: EmailStr
    password: str
    confirm_password: str
    agreed_terms: bool

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class AdminStats(BaseModel):
    total_users: int
    total_products: int

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    agreed_terms: bool
    created_at: datetime
    phone: str | None = None
    ubicacion: str | None = None
    district_id: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    occupation: str | None = None
    bio: str | None = None
    dni: str | None = None
    credits: int
    interests: List[str] = []
    profile_picture: str | None = None
    is_active: bool

    rating_score: Optional[float] = 0.0
    rating_count: int = 0

    role: str = 'user'

    class Config:
        from_attributes = True

# --- Modelos Pydantic para JWT ---
class Token(BaseModel):
    access_token: str
    token_type: str

class ProposalStatusUpdate(BaseModel):
    status: Literal['accepted', 'rejected', 'cancelled', 'completed']

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None

class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str
    confirm_new_password: str

class UserRating(Base):
    __tablename__ = 'user_ratings'
    id = Column(Integer, primary_key=True, index=True)
    rater_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    rated_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    proposal_id = Column(Integer, ForeignKey('proposals.id', ondelete='CASCADE'), nullable=False)
    score = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    rater = relationship("User", foreign_keys=[rater_id], back_populates="ratings_given")
    rated_user = relationship("User", foreign_keys=[rated_id], back_populates="ratings_received")
    proposal = relationship("Proposal")



# --- Configuración JWT ---
SECRET_KEY = os.getenv("SECRET_KEY", "your_super_secret_key_that_you_should_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None or user_id is None:
            raise credentials_exception
        token_data = TokenData(email=email, user_id=user_id)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    """
    Una dependencia que obtiene el usuario actual y verifica si tiene el rol 'admin'.
    Si no es admin, lanza un error HTTP 403.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permisos de administrador para realizar esta acción."
        )
    return current_user
    
class ProductImageResponse(BaseModel):
    id: int
    product_id: int
    image_url: str
    is_thumbnail: bool
    upload_order: Optional[int]
    uploaded_at: datetime

    class Config:
        from_attributes = True

class CategoryResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True
        
class ProductResponse(BaseModel):
    id: int
    user_id: int
    category_id: int
    title: str
    description: str
    current_value_estimate: Optional[float]
    condition: str
    status: str
    preffered_exchange_items: Optional[str]
    location: Optional[str]
    is_active: bool
    views_count: int
    created_at: datetime
    updated_at: datetime
    
    user_username: Optional[str] = None
    user_avatar_url: Optional[str] = None
    category_name: Optional[str] = None
    thumbnail_image_url: Optional[str] = None
    images: List[ProductImageResponse] = []
    exchange_interests: List[str] = []

    class Config:
        from_attributes = True


class UserPublicResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    bio: Optional[str] = None 
    ubicacion: Optional[str] = None
    interests: List[CategoryResponse] = []
    avatar: Optional[str] = None # Campo para simular el avatar en el frontend

    rating_score: Optional[float] = 0.0
    rating_count: int = 0

    class Config:
        from_attributes = True
        populate_by_name = True

# Esquema Pydantic para ProductBasicInfo (información básica del producto) - Renombrado/Ajustado a ProductPublicResponse
class ProductPublicResponse(BaseModel): # Un modelo más ligero para productos en el feed/proposals
    id: int
    title: str
    description: str
    thumbnail_image_url: Optional[str] = None

    class Config:
        from_attributes = True

class ChartDataPoint(BaseModel):
    label: str  # La etiqueta del eje X (ej. '2023-10-27' o '2023-W43')
    count: int  # El valor del eje Y

class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, object]]

# Esquema Pydantic para ProposalCreate
class ProposalCreate(BaseModel):
    offered_product_id: int
    requested_product_id: int

# Esquema Pydantic para ProposalResponse (con info de usuarios y productos anidados) - Tu modelo original
class ProposalResponse(BaseModel):
    id: int
    offered_product_id: int
    requested_product_id: int
    proposer_user_id: int
    owner_of_requested_product_id: int
    status: str
    created_at: datetime
    updated_at: datetime

    offered_product: ProductPublicResponse # Usar el modelo público
    requested_product: ProductPublicResponse # Usar el modelo público
    proposer: UserPublicResponse # Usar el modelo público
    owner_of_requested_product: UserPublicResponse # Usar el modelo público

    class Config:
        from_attributes = True

class ReportCreate(BaseModel):
    reason: str

class HeroData(BaseModel):
    titleLine1: str
    titleLine2: str
    badgeText: str      
    paragraphText: str  
    button1Text: str    
    button2Text: str    
    cards: List[HeroCard] = []

class MessageResponse(BaseModel):
    id: int
    proposal_id: int
    sender_id: int
    text: Optional[str] = None # <-- MODIFICADO: Ahora es opcional
    image_url: Optional[str] = None # <-- NUEVO: Para enviar la URL en la respuesta
    timestamp: datetime
    is_read: bool

    class Config:
        from_attributes = True
    
class ProductUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    status: Optional[str] = None # Para cambiar el estado (disponible, intercambiado, etc.)
    category_name: Optional[str] = None
    is_for_sale: Optional[bool] = None 


# NUEVO: Pydantic model para la información de intercambio dentro de una conversación
class ExchangeDetailsResponse(BaseModel):
    id: int
    offer: ProductPublicResponse
    request: ProductPublicResponse
    created_at: datetime  # <-- Cambio aplicado aquí
    status: str
    proposer_user_id: int

    class Config:
        from_attributes = True


# NUEVO: Pydantic model para una "conversación" (lo que espera tu InboxView.vue)
class ConversationResponse(BaseModel):
    exchange: ExchangeDetailsResponse
    user: UserPublicResponse # El "otro" usuario en la conversación
    messages: List[MessageResponse]
    last_message: MessageResponse # El último mensaje de la conversación
    unread_count: int = 0 # Mensajes no leídos para el usuario actual

    class Config:
        from_attributes = True

def get_current_user_from_token(token: str, db: Session) -> Optional[User]:
    """
    Decodifica un token JWT y devuelve el usuario si es válido.
    Funciona para WebSockets donde Depends() no se puede usar directamente.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            return None
        # Busca el usuario en la base de datos
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except JWTError:
        return None


@app.get("/")
async def root():
    return {"message": "¡Bienvenido a la API de KambiaPe! ✅"} 

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # 1. Validación de contraseñas
    if user.password != user.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Las contraseñas no coinciden."
        )

    # 2. VALIDACIÓN DE DNI Y OBTENCIÓN DE DATOS COMPLETOS
    # Llama a la función y obtiene los datos verificados del API
    api_data = validate_dni_data(dni=user.dni, full_name=user.full_name) 

    # 3. Verificación de existencia de DNI
    existing_dni_user = db.query(User).filter(User.dni == user.dni).first()
    if existing_dni_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El número de DNI ya está registrado por otro usuario."
        )
        
    # 4. Verificación de existencia de email
    existing_email_user = db.query(User).filter(User.email == user.email).first()
    if existing_email_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El correo electrónico ya está registrado."
        )

    # 5. Creación de usuario (usando datos del API)
    hashed_password = get_password_hash(user.password)
    
    # --- PROCESAR DATOS OBTENIDOS DEL API COMPLETO ---
    # Convertir la fecha de nacimiento de la API ("DD/MM/YYYY") a objeto date de Python.
    fecha_nacimiento_api = None
    if 'fecha_nacimiento' in api_data:
        try:
            fecha_nacimiento_api = datetime.strptime(api_data['fecha_nacimiento'], '%d/%m/%Y').date()
        except ValueError:
            # Si el formato es incorrecto, lo dejamos como None
            pass
    
    # ===== INICIO DE LA MODIFICACIÓN: VERIFICACIÓN DE EDAD =====
    if not fecha_nacimiento_api:
        # Si el API de DNI no nos dio una fecha, no podemos verificar la edad.
        # Por política de seguridad, rechazamos el registro.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo verificar tu fecha de nacimiento desde el DNI. No es posible completar el registro."
        )

    today = datetime.utcnow().date()
    # Calculamos la fecha límite (exactamente 18 años atrás)
    eighteen_years_ago = today.replace(year=today.year - 18)
    
    # Si la fecha de nacimiento es MÁS RECIENTE que la fecha límite, es menor de 18.
    if fecha_nacimiento_api > eighteen_years_ago:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debes ser mayor de 18 años para registrarte en KambiaPe."
        )
    # ===== FIN DE LA MODIFICACIÓN =====
        
    genero_api = api_data.get('genero') 
    
    # Tomamos el nombre completo verificado, la fecha de nacimiento y el género del API
    db_user = User(
        full_name=api_data['nombre_completo'], # USAMOS EL NOMBRE VERIFICADO
        email=user.email,
        dni=user.dni,
        password_hash=hashed_password,
        agreed_terms=user.agreed_terms,
        # GUARDAMOS LOS NUEVOS CAMPOS
        date_of_birth=fecha_nacimiento_api,
        gender=genero_api,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return UserResponse.from_orm(db_user)

@app.post("/login", response_model=Token)
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_login.email).first()

    if not db_user or not verify_password(user_login.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credenciales incorrectas: Correo o contraseña inválidos."
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email, "user_id": db_user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/ratings", response_model=UserRatingResponse, status_code=status.HTTP_201_CREATED) # <-- CAMBIO AQUÍ
def create_rating(
    rating: UserRatingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Verificar que la propuesta exista y esté aceptada
    proposal = db.query(Proposal).filter(
        Proposal.id == rating.proposal_id,
        Proposal.status == 'accepted'
    ).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada o no ha sido aceptada.")

    # 2. Verificar que el usuario actual sea parte de la propuesta y no sea el calificado
    if not (proposal.proposer_user_id == current_user.id or proposal.owner_of_requested_product_id == current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No eres parte de esta propuesta.")
    
    if rating.rated_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No te puedes calificar a ti mismo.")

    # 3. Verificar que el usuario calificado sea la contraparte en la propuesta
    is_counterpart = (
        (proposal.proposer_user_id == current_user.id and proposal.owner_of_requested_product_id == rating.rated_id) or
        (proposal.owner_of_requested_product_id == current_user.id and proposal.proposer_user_id == rating.rated_id)
    )
    if not is_counterpart:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El usuario calificado no es la contraparte en esta propuesta.")

    # 4. Verificar que no exista ya una calificación para esta propuesta por este usuario
    existing_rating = db.query(UserRating).filter(
        UserRating.proposal_id == rating.proposal_id,
        UserRating.rater_id == current_user.id
    ).first()

    if existing_rating:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya has calificado esta transacción.")

    # 5. Crear la valoración
    db_rating = UserRating(
        **rating.dict(),
        rater_id=current_user.id
    )
    db.add(db_rating)
    db.commit()
    db.refresh(db_rating)
    return db_rating


@app.put("/proposals/{proposal_id}/status", response_model=ProposalResponse)
async def update_proposal_status(
    proposal_id: int,
    status_update: ProposalStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Actualiza el estado de una propuesta y el de los productos involucrados.
    - 'accepted': Cambia los productos a 'pending_exchange'.
    - 'rejected': No cambia el estado de los productos.
    - 'cancelled': Devuelve los productos a 'available'.
    - 'completed': Cambia los productos a 'exchanged'.
    """
    proposal = db.query(Proposal).options(
        joinedload(Proposal.offered_product),
        joinedload(Proposal.requested_product),
        joinedload(Proposal.proposer),
        joinedload(Proposal.owner_of_requested_product)
    ).filter(Proposal.id == proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")

    new_status = status_update.status
    user_id = current_user.id

    is_participant = user_id in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]
    if not is_participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No eres parte de esta propuesta.")

    # Lógica de permisos para aceptar o rechazar
    if new_status in ['accepted', 'rejected']:
        if user_id != proposal.owner_of_requested_product_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para aceptar o rechazar esta propuesta.")
        if proposal.status != 'pending':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No se puede {new_status} una propuesta que no está pendiente.")
        
        # --- LÓGICA DE ESTADO DE PRODUCTO ---
        if new_status == 'accepted':
            # Si se acepta, ambos productos pasan a estar "En Intercambio"
            proposal.offered_product.status = 'pending_exchange'
            proposal.requested_product.status = 'pending_exchange'
            db.add(proposal.offered_product)
            db.add(proposal.requested_product)

    # Lógica de permisos para cancelar
    elif new_status == 'cancelled':
        if user_id != proposal.proposer_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No puedes cancelar una propuesta que no hiciste.")
        if proposal.status not in ['pending', 'accepted']:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se puede cancelar esta propuesta.")
        
        # --- LÓGICA DE ESTADO DE PRODUCTO ---
        # Si se cancela, ambos productos vuelven a estar "Disponibles"
        proposal.offered_product.status = 'available'
        proposal.requested_product.status = 'available'
        db.add(proposal.offered_product)
        db.add(proposal.requested_product)

    # Lógica para completar la propuesta
    elif new_status == 'completed':
        if proposal.status != 'accepted':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Solo se puede completar una propuesta que ha sido aceptada.")
        
        # --- LÓGICA DE ESTADO DE PRODUCTO ---
        # Si se completa, ambos productos se marcan como "Intercambiados"
        proposal.offered_product.status = 'exchanged'
        proposal.requested_product.status = 'exchanged'
        db.add(proposal.offered_product)
        db.add(proposal.requested_product)

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Estado no válido.")

    # Si pasa todas las validaciones, se actualiza el estado de la propuesta
    proposal.status = new_status
    db.commit()
    db.refresh(proposal)
    
    return proposal

@app.get("/admin/stats", response_model=AdminStats)
async def get_admin_stats(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user) # <-- ¡Esto lo protege!
):
    """
    Obtiene estadísticas básicas para el dashboard de admin.
    Solo accesible por administradores.
    """
    try:
        # Cuenta el total de usuarios en la tabla User
        total_users = db.query(User).count()
        
        # Cuenta el total de productos en la tabla Product
        total_products = db.query(Product).count()
        
        return AdminStats(
            total_users=total_users, 
            total_products=total_products
        )
    except Exception as e:
        print(f"Error al calcular estadísticas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener las estadísticas."
        )

@app.get("/profile/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Verifica que el usuario solo pueda ver su propio perfil.
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para ver este perfil.")
    
    # Carga el usuario y sus relaciones de intereses y valoraciones de forma eficiente
    user = db.query(User).options(
        joinedload(User.interests),
        joinedload(User.ratings_received) # <-- Se añade la carga de valoraciones
    ).filter(User.id == user_id).first()
    
    # Si el usuario no existe, devuelve un error 404
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # Lógica para calcular el rating
    rating_count = len(user.ratings_received)
    if rating_count > 0:
        rating_score = sum(r.score for r in user.ratings_received) / rating_count
    else:
        rating_score = 0.0

    # Construye el diccionario de respuesta, asegurándose de incluir todos los campos.
    user_data = {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "agreed_terms": user.agreed_terms,
        "created_at": user.created_at,
        "phone": user.phone,
        "ubicacion": user.ubicacion,
        "district_id": user.district_id,
        "date_of_birth": user.date_of_birth,
        "gender": user.gender,
        "occupation": user.occupation,
        "bio": user.bio,
        "dni": user.dni,
        "credits": user.credits,
        "interests": [interest.name for interest in user.interests],
        "profile_picture": user.profile_picture,
        "rating_score": rating_score,    
        "rating_count": rating_count, 
        "role": user.role,
        "is_active": user.is_active  
    }
    
    # Devuelve la respuesta usando el modelo Pydantic UserResponse
    return UserResponse(**user_data)

class UserUpdate(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    ubicacion: str | None = None
    district_id: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    occupation: str | None = None
    bio: str | None = None
    dni: str | None = None
    interest_ids: Optional[List[int]] = []

@app.get("/proposals/{proposal_id}/messages", response_model=List[MessageResponse])
async def get_proposal_messages(
    proposal_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")
    
    if current_user.id not in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para ver estos mensajes.")

    db.query(Message).filter(
        Message.proposal_id == proposal_id,
        Message.sender_id != current_user.id,
        Message.is_read == False
    ).update({"is_read": True})
    db.commit()

    messages = db.query(Message).filter(Message.proposal_id == proposal_id).order_by(Message.timestamp.asc()).all()
    return messages

@app.get("/api/admin/users", response_model=List[UserResponse])
async def get_all_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user) # <-- Protección de admin
):
    """
    Obtiene una lista de todos los usuarios. Solo para administradores.
    """
    # Usamos joinedload para cargar las relaciones que UserResponse necesita
    users = db.query(User).options(
        joinedload(User.interests),
        joinedload(User.ratings_received)
    ).order_by(User.id.asc()).all()

    # Debemos construir manualmente la respuesta para
    # asegurarnos de que los campos 'rating_score' y 'rating_count' se calculen
    user_responses = []
    for user in users:
        rating_count = len(user.ratings_received)
        if rating_count > 0:
            rating_score = sum(r.score for r in user.ratings_received) / rating_count
        else:
            rating_score = 0.0

        user_data = UserResponse(
            id=user.id,
            full_name=user.full_name,
            email=user.email,
            agreed_terms=user.agreed_terms,
            created_at=user.created_at,
            phone=user.phone,
            ubicacion=user.ubicacion,
            district_id=user.district_id,
            date_of_birth=user.date_of_birth,
            gender=user.gender,
            occupation=user.occupation,
            bio=user.bio,
            dni=user.dni,
            credits=user.credits,
            interests=[interest.name for interest in user.interests],
            profile_picture=user.profile_picture,
            rating_score=rating_score,
            rating_count=rating_count, 
            role=user.role,
            is_active=user.is_active
        )
        user_responses.append(user_data)
    
    return user_responses

@app.get("/categories", response_model=List[CategoryResponse])
async def get_all_categories(db: Session = Depends(get_db)):
    categories = db.query(Category).order_by(Category.name).all()
    return categories

@app.put("/profile/{user_id}", response_model=UserResponse)
async def update_user_profile(user_id: int, user_data: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para actualizar este perfil.")

    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    update_data = user_data.model_dump(exclude_unset=True)

    if 'interest_ids' in update_data:
        interest_ids = update_data.pop('interest_ids')
        if interest_ids is not None:
            interests = db.query(Category).filter(Category.id.in_(interest_ids)).all()
            db_user.interests = interests
        else:
            db_user.interests = []

    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return await get_user_profile(user_id, db, db_user)

@app.post("/profile/picture", response_model=UserResponse)
async def upload_profile_picture(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    image_url = await save_upload_file(file)
    current_user.profile_picture = image_url
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return await get_user_profile(current_user.id, db, current_user)

@app.put("/users/change-password", status_code=status.HTTP_200_OK)
async def change_user_password(
    password_data: PasswordUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if password_data.new_password != password_data.confirm_new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La nueva contraseña y la confirmación no coinciden.")
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="La contraseña actual es incorrecta.")
    
    new_hashed_password = get_password_hash(password_data.new_password)
    current_user.password_hash = new_hashed_password
    db.add(current_user)
    db.commit()
    return {"message": "Contraseña actualizada exitosamente."}


@app.get("/users/{user_id}/products", response_model=List[ProductResponse])
async def get_user_products(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Se eliminó la validación 'if user_id != current_user.id:' para permitir la visualización.
    # La dependencia 'current_user' se mantiene para asegurar que solo usuarios autenticados puedan acceder.

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    products_db = db.query(Product).options(
        joinedload(Product.category_obj),
        joinedload(Product.images),
        joinedload(Product.exchange_interests)
    ).filter(Product.user_id == user_id, Product.is_active == True).order_by(Product.created_at.desc()).all()

    response_products = []
    for product in products_db:
        thumbnail_url = None
        if product.images:
            # Busca una imagen thumbnail o usa la primera si no hay ninguna.
            thumbnail_image = next((img for img in product.images if img.is_thumbnail), product.images[0])
            thumbnail_url = thumbnail_image.image_url
        
        response_products.append(ProductResponse(
            id=product.id,
            user_id=product.user_id,
            category_id=product.category_id,
            title=product.title,
            description=product.description,
            current_value_estimate=product.current_value_estimate,
            condition=product.condition,
            status=product.status,
            preffered_exchange_items=product.preffered_exchange_items,
            location=product.location,
            is_active=product.is_active,
            views_count=product.views_count,
            created_at=product.created_at,
            updated_at=product.updated_at,
            category_name=product.category_obj.name if product.category_obj else "Sin categoría",
            thumbnail_image_url=thumbnail_url,
            images=[ProductImageResponse.from_orm(img) for img in product.images],
            exchange_interests=[interest.name for interest in product.exchange_interests],
            user_username=user.full_name
        ))
    return response_products

@app.on_event("startup")
async def startup_event():
    await broadcast.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await broadcast.disconnect()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    # --- Gestión de sesión para la conexión inicial ---
    # Ya no usamos Depends(get_db) aquí.
    db = SessionLocal()
    try:
        current_user = db.query(User).filter(User.id == user_id).first()
        if not current_user:
            await websocket.accept() # Aceptar para enviar el código de cierre
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Guardamos el ID del usuario verificado.
        user_id_safe = current_user.id
    finally:
        db.close() # Cerramos esta sesión inicial
    
    await websocket.accept()
    # --- Fin de la gestión de sesión inicial ---

    # Creamos una tarea para escuchar mensajes entrantes y otra para enviar mensajes salientes
    async def receive_messages():
        try:
            while True:
                data = await websocket.receive_text()
                message_data = json.loads(data)

                proposal_id = message_data.get('proposal_id')
                if not proposal_id:
                    continue # Ignorar eventos sin proposal_id

                event_type = message_data.get('type')
                
                # ===================== INICIO DEL CAMBIO CLAVE =====================
                # Creamos una sesión de BD CORTA para CADA mensaje/evento
                db = SessionLocal()
                try:
                    # --- 1. Validar la propuesta y encontrar al destinatario ---
                    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

                    # Verificar que el usuario actual pertenece a esta conversación
                    if not proposal or (user_id_safe not in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]):
                        continue

                    # Identificar al otro usuario (destinatario)
                    recipient_id = proposal.owner_of_requested_product_id if user_id_safe == proposal.proposer_user_id else proposal.proposer_user_id
                    
                    # --- 2. Manejar el evento según su tipo ---
                    
                    if event_type == 'typing_start':
                        # Este evento NO necesita la BD, solo retransmite
                        payload = json.dumps({
                            "type": "user_typing",
                            "data": {"proposal_id": proposal_id, "user_id": user_id_safe}
                        })
                        await broadcast.publish(channel=f"user_{recipient_id}", message=payload)

                    elif event_type == 'typing_stop':
                        # Este evento NO necesita la BD, solo retransmite
                        payload = json.dumps({
                            "type": "user_stopped_typing",
                            "data": {"proposal_id": proposal_id, "user_id": user_id_safe}
                        })
                        await broadcast.publish(channel=f"user_{recipient_id}", message=payload)

                    elif event_type == 'new_message' or not event_type:
                        text_content = message_data.get('text')
                        if not text_content:
                            continue # No guardar mensajes vacíos

                        # --- 3. Guardar el nuevo mensaje en la BD (usando la sesión CORTA) ---
                        db_message = Message(
                            proposal_id=proposal_id,
                            sender_id=user_id_safe,
                            text=text_content,
                            timestamp=datetime.utcnow()
                        )
                        db.add(db_message)
                        db.commit()
                        db.refresh(db_message)
                        
                        # --- 4. Preparar y enviar el mensaje real ---
                        message_data_dict = MessageResponse.from_orm(db_message).model_dump()
                        
                        payload = json.dumps({
                            "type": "new_message",
                            "data": message_data_dict
                        }, default=str)

                        # Publica el mensaje en el canal del destinatario Y en el del remitente
                        await broadcast.publish(channel=f"user_{recipient_id}", message=payload)
                        await broadcast.publish(channel=f"user_{user_id_safe}", message=payload)
                
                finally:
                    db.close() # MUY IMPORTANTE: Cerramos la sesión CORTA
                # ====================== FIN DEL CAMBIO CLAVE =======================
        
        except WebSocketDisconnect:
            pass # El bucle de subscribe se encargará de la desconexión
        except Exception as e:
            print(f"Error en receive_messages para usuario {user_id_safe}: {e}")

    async def send_messages():
        # Esta función no cambia. Solo escucha y envía lo que recibe.
        try:
            async with broadcast.subscribe(channel=f"user_{user_id_safe}") as subscriber:
                async for event in subscriber:
                    await websocket.send_text(event.message)
        except Exception as e:
            print(f"Error en send_messages para usuario {user_id_safe}: {e}")

    # Ejecuta ambas tareas concurrentemente
    try:
        await asyncio.gather(receive_messages(), send_messages())
    except Exception as e:
        print(f"WebSocket cerrado para el usuario {user_id_safe}. Error: {e}")

@app.get("/users/{user_id}/public-profile", response_model=UserPublicResponse)
async def get_public_user_profile(user_id: int, db: Session = Depends(get_db)):
    """
    Obtiene el perfil público de un usuario, incluyendo su valoración.
    Este endpoint es público y no requiere autenticación del solicitante.
    """
    user = db.query(User).options(
        joinedload(User.interests),
        joinedload(User.ratings_received)
    ).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # Lógica para calcular el rating (la misma que usas en el perfil privado)
    rating_count = len(user.ratings_received)
    if rating_count > 0:
        rating_score = sum(r.score for r in user.ratings_received) / rating_count
    else:
        rating_score = 0.0

    # Construye la respuesta usando el modelo Pydantic UserPublicResponse
    return UserPublicResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        bio=user.bio,
        ubicacion=user.ubicacion,
        interests=[CategoryResponse.from_orm(interest) for interest in user.interests],
        avatar=user.profile_picture,
        rating_score=rating_score,
        rating_count=rating_count
    )

@app.get("/admin/all_reports", response_model=List[UserReportResponse])
async def get_all_user_reports(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user) # <--- PROTEGIDO
):
    """
    Obtiene todos los reportes de usuarios. Solo para administradores.
    """
    reports = db.query(UserReport).order_by(UserReport.created_at.desc()).all()
    return reports

@app.delete("/proposals/{proposal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_proposal(
    proposal_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Realiza un borrado lógico de la conversación para el usuario actual.
    No elimina el registro, solo lo marca como borrado para uno de los participantes.
    """
    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

    if not proposal:
        # No es necesario un error aquí, si no la encuentra, para el usuario es como si ya estuviera borrada.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Identificar el rol del usuario actual en esta propuesta
    if proposal.proposer_user_id == current_user.id:
        # El usuario que borra es el que hizo la propuesta
        proposal.deleted_by_proposer = True
    elif proposal.owner_of_requested_product_id == current_user.id:
        # El usuario que borra es el que recibió la propuesta
        proposal.deleted_by_receiver = True
    else:
        # Si no es ninguno de los dos, no tiene permiso
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para modificar esta conversación."
        )
    
    db.commit() # Guarda el cambio (el flag de borrado)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/products_feed", response_model=List[ProductResponse])
async def get_all_products_for_feed(db: Session = Depends(get_db)):
    products_db = db.query(Product).options(
        joinedload(Product.category_obj),
        joinedload(Product.images),
        joinedload(Product.owner),
        joinedload(Product.exchange_interests) # <-- ✨ CORRECCIÓN CLAVE
    ).filter(Product.is_active == True, Product.status.in_(['available', 'pending_exchange'])).all()

    response_products = []
    for product in products_db:
        thumbnail_url = None
        if product.images:
            thumbnail_image = next((img for img in product.images if img.is_thumbnail), None)
            if thumbnail_image:
                thumbnail_url = thumbnail_image.image_url
            else:
                thumbnail_url = product.images[0].image_url

        response_products.append(ProductResponse(
            id=product.id, user_id=product.user_id, category_id=product.category_id, title=product.title,
            description=product.description, current_value_estimate=product.current_value_estimate,
            condition=product.condition, status=product.status, preffered_exchange_items=product.preffered_exchange_items,
            location=product.location, is_active=product.is_active, views_count=product.views_count,
            created_at=product.created_at, updated_at=product.updated_at,
            category_name=product.category_obj.name if product.category_obj else "Sin categoría",
            thumbnail_image_url=thumbnail_url, 
            exchange_interests=[interest.name for interest in product.exchange_interests],
            images=[ProductImageResponse.from_orm(img) for img in product.images],
            user_username=product.owner.full_name if product.owner else "Anónimo",
            user_avatar_url=product.owner.profile_picture if product.owner else None
        ))
    return response_products

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        async with aiofiles.open(file_path, "wb") as buffer:
            while content := await upload_file.read(1024 * 1024):
                await buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"No se pudo guardar la imagen: {upload_file.filename}")
    return f"/{UPLOAD_DIR}/{unique_filename}"


@app.post("/products", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    title: str = Form(...),
    description: str = Form(...),
    category_name: str = Form(...),
    condition: str = Form(...),
    current_value_estimate: Optional[float] = Form(None),
    preffered_exchange_items: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    photos: List[UploadFile] = File(...),
    exchange_interest_ids: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # --- ✨ 1. VALIDACIÓN DE PERFIL COMPLETO (MODIFICACIÓN AÑADIDA) ---
    # Se comprueba si el usuario tiene teléfono y ubicación/distrito.
    if not current_user.phone or not current_user.ubicacion or not current_user.district_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debes completar tu información de perfil (teléfono y ubicación) antes de poder publicar un producto."
        )
    # --- FIN DE LA VALIDACIÓN DE PERFIL ---

    # --- 2. VERIFICACIÓN Y DESCUENTO DE CRÉDITOS ---
    # Se comprueba si el usuario tiene créditos suficientes.
    if current_user.credits is None or current_user.credits < 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes suficientes créditos para publicar."
        )

    # Si tiene créditos, se le descuenta uno.
    current_user.credits -= 1
    db.add(current_user) # Se añade el cambio a la sesión de la BD.
    # --- FIN DE LA LÓGICA DE CRÉDITOS ---

    # --- 3. VALIDACIONES DEL PRODUCTO ---
    if not all([title.strip(), description.strip(), category_name.strip(), condition.strip()]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Los campos de título, descripción, categoría y condición son obligatorios.")
    if not photos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Se requiere al menos una imagen.")
    if len(photos) > 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Solo se permiten un máximo de 4 imágenes.")

    # --- 4. LÓGICA DE NEGOCIO (Categorías e Intereses) ---
    category = db.query(Category).filter(Category.name == category_name).first()
    if not category:
        # Si la categoría no existe, se revierte el descuento de crédito para no penalizar al usuario.
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"La categoría '{category_name}' no existe.")

    interest_ids_list = []
    if exchange_interest_ids:
        try:
            interest_ids_list = [int(id_str) for id_str in exchange_interest_ids.split(',') if id_str]
        except ValueError:
            db.rollback() # Revertir descuento de crédito si hay error
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Formato inválido para IDs de intereses.")

    # --- 5. CREACIÓN DEL PRODUCTO Y GUARDADO DE IMÁGENES ---
    new_product = Product(
        user_id=current_user.id, category_id=category.id, title=title, description=description,
        current_value_estimate=current_value_estimate, condition=condition,
        preffered_exchange_items=preffered_exchange_items, location=location,
    )
    db.add(new_product)
    db.flush() # Se asigna un ID al producto antes de guardar las imágenes

    for i, photo in enumerate(photos):
        try:
            # Asumo que 'save_upload_file' es una función que has creado para guardar archivos
            image_url = await save_upload_file(photo) 
            db.add(ProductImage(product_id=new_product.id, image_url=image_url, is_thumbnail=(i == 0), upload_order=i + 1))
        except Exception:
            db.rollback() # Si falla la subida de una imagen, se revierte todo
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al subir la imagen {photo.filename}.")

    # --- 6. ASIGNACIÓN DE INTERESES Y COMMIT FINAL ---
    if interest_ids_list:
        interests_to_add = db.query(Category).filter(Category.id.in_(interest_ids_list)).all()
        if interests_to_add:
            new_product.exchange_interests = interests_to_add

    try:
        # Se confirman todos los cambios en la base de datos:
        # el descuento de crédito, el nuevo producto y sus imágenes e intereses.
        db.commit()
        db.refresh(new_product)
    except Exception as e:
        db.rollback() # Si el commit final falla, se revierte todo
        raise HTTPException(status_code=500, detail=f"Error al guardar el producto en la base de datos: {e}")
        
    # --- 7. CONSTRUCCIÓN DE LA RESPUESTA ---
    thumbnail_url = None
    if new_product.images:
        # Busca la imagen thumbnail o usa la primera si no hay ninguna marcada como tal.
        thumb = next((img for img in new_product.images if img.is_thumbnail), new_product.images[0])
        thumbnail_url = thumb.image_url

    return ProductResponse(
        id=new_product.id, user_id=new_product.user_id, category_id=new_product.category_id,
        title=new_product.title, description=new_product.description, current_value_estimate=new_product.current_value_estimate,
        condition=new_product.condition, status=new_product.status, preffered_exchange_items=new_product.preffered_exchange_items,
        location=new_product.location, is_active=new_product.is_active, views_count=new_product.views_count,
        created_at=new_product.created_at, updated_at=new_product.updated_at,
        user_username=new_product.owner.full_name, # Asegúrate de que la relación se llama 'owner'
        category_name=new_product.category_obj.name, # Asegúrate de que la relación se llama 'category_obj'
        thumbnail_image_url=thumbnail_url,
        images=[ProductImageResponse.from_orm(img) for img in new_product.images],
        exchange_interests=[interest.name for interest in new_product.exchange_interests]
    )

# Pega esto en main.py

# Pega esto en main.py (reemplazando tu función /api/admin/login existente)

@app.post("/api/admin/login", response_model=AdminLoginResponse) # <-- Usa el nuevo response_model
async def login_admin(user_login: UserLogin, db: Session = Depends(get_db)):
    
    # 1. Verificamos las credenciales (igual que en el login normal)
    db_user = db.query(User).options(joinedload(User.interests)).filter(User.email == user_login.email).first()

    if not db_user or not verify_password(user_login.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credenciales incorrectas: Correo o contraseña inválidos."
        )
    
    # 2. ¡VERIFICACIÓN DE ROL! (Esta es la parte clave)
    if db_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado. No tienes permisos de administrador."
        )

    # 3. Si es admin, creamos y devolvemos el token + usuario
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token_data = {
        "sub": db_user.email, 
        "user_id": db_user.id,
        "role": db_user.role  # Incluimos el rol en el token
    }

    access_token = create_access_token(
        data=access_token_data,
        expires_delta=access_token_expires
    )
    
    # Calculamos el rating (ya que UserResponse lo requiere)
    rating_count = len(db_user.ratings_received)
    rating_score = 0.0
    if rating_count > 0:
        rating_score = sum(r.score for r in db_user.ratings_received) / rating_count

    # Construimos la respuesta de UserResponse
    user_response_data = UserResponse(
        id=db_user.id,
        full_name=db_user.full_name,
        email=db_user.email,
        agreed_terms=db_user.agreed_terms,
        created_at=db_user.created_at,
        phone=db_user.phone,
        ubicacion=db_user.ubicacion,
        district_id=db_user.district_id,
        date_of_birth=db_user.date_of_birth,
        gender=db_user.gender,
        occupation=db_user.occupation,
        bio=db_user.bio,
        dni=db_user.dni,
        credits=db_user.credits,
        interests=[interest.name for interest in db_user.interests],
        profile_picture=db_user.profile_picture,
        rating_score=rating_score,
        rating_count=rating_count,
        is_active=db_user.is_active,
        
        # --- ESTA ES LA LÍNEA QUE SOLUCIONA EL PROBLEMA ---
        role=db_user.role
        # ----------------------------------------------------
    )

    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user_response_data # Ahora 'user' SÍ incluye 'is_admin: true'
    }

@app.post("/proposals", response_model=ProposalResponse, status_code=status.HTTP_201_CREATED)
async def create_proposal(
    proposal: ProposalCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    offered_product = db.query(Product).filter(Product.id == proposal.offered_product_id, Product.user_id == current_user.id).first()
    if not offered_product:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El producto ofrecido no existe o no te pertenece.")
    
    requested_product = db.query(Product).filter(Product.id == proposal.requested_product_id).first()
    if not requested_product:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El producto solicitado no existe.")
    if requested_product.user_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No puedes solicitar un intercambio por tu propio producto.")
    
    existing_proposal = db.query(Proposal).filter(
        Proposal.offered_product_id == proposal.offered_product_id,
        Proposal.requested_product_id == proposal.requested_product_id,
        Proposal.proposer_user_id == current_user.id,
        Proposal.status == "pending"
    ).first()
    if existing_proposal:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya existe una propuesta pendiente para este intercambio.")

    new_proposal = Proposal(
        offered_product_id=proposal.offered_product_id,
        requested_product_id=proposal.requested_product_id,
        proposer_user_id=current_user.id,
        owner_of_requested_product_id=requested_product.user_id,
        status="pending"
    )
    db.add(new_proposal)
    db.commit()
    db.refresh(new_proposal)

    db_proposal = db.query(Proposal).options(
        joinedload(Proposal.offered_product),
        joinedload(Proposal.requested_product),
        joinedload(Proposal.proposer),
        joinedload(Proposal.owner_of_requested_product)
    ).filter(Proposal.id == new_proposal.id).first()
    return db_proposal

@app.post("/users/{user_id_to_block}/block", status_code=status.HTTP_204_NO_CONTENT)
async def block_user(
    user_id_to_block: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if user_id_to_block == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No te puedes bloquear a ti mismo.")

    user_to_block = db.query(User).filter(User.id == user_id_to_block).first()
    if not user_to_block:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # Verificar si ya está bloqueado para no duplicar
    if user_to_block not in current_user.blocked_users:
        current_user.blocked_users.append(user_to_block)
        db.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/users/{user_id_to_report}/report", status_code=status.HTTP_201_CREATED)
async def report_and_block_user(
    user_id_to_report: int,
    report_data: ReportCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if user_id_to_report == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No te puedes reportar a ti mismo.")

    user_to_report = db.query(User).filter(User.id == user_id_to_report).first()
    if not user_to_report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # Crear el reporte
    new_report = UserReport(
        reporter_id=current_user.id,
        reported_id=user_id_to_report,
        reason=report_data.reason
    )
    db.add(new_report)

    # Bloquear al usuario (si no está ya bloqueado)
    if user_to_report not in current_user.blocked_users:
        current_user.blocked_users.append(user_to_report)
    
    db.commit()
    
    return {"message": "Usuario reportado y bloqueado con éxito."}

@app.get("/proposals/me", response_model=List[ConversationResponse])
async def get_my_proposals(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Obtener una lista de IDs de usuarios que el usuario actual ha bloqueado
    #    y también los que lo han bloqueado a él.
    blocked_user_ids = {user.id for user in current_user.blocked_users}
    users_who_blocked_current_user = {user.id for user in current_user.blocked_by_users}
    all_blocked_ids = blocked_user_ids.union(users_who_blocked_current_user)

    # 2. La consulta principal ahora excluye a los usuarios bloqueados
    proposals = db.query(Proposal).options(
        joinedload(Proposal.offered_product).joinedload(Product.images),
        joinedload(Proposal.requested_product).joinedload(Product.images),
        joinedload(Proposal.proposer).joinedload(User.interests),
        joinedload(Proposal.owner_of_requested_product).joinedload(User.interests),
        joinedload(Proposal.messages).joinedload(Message.sender_obj)
    ).filter(
        or_(
            and_(Proposal.proposer_user_id == current_user.id, Proposal.deleted_by_proposer == False),
            and_(Proposal.owner_of_requested_product_id == current_user.id, Proposal.deleted_by_receiver == False)
        ),
        # 3. Añadir este filtro para excluir conversaciones con usuarios bloqueados
        Proposal.proposer_user_id.notin_(all_blocked_ids),
        Proposal.owner_of_requested_product_id.notin_(all_blocked_ids)
    ).order_by(Proposal.updated_at.desc()).all()

    conversations_data = []
    for proposal in proposals:
        other_user_obj = proposal.owner_of_requested_product if proposal.proposer_user_id == current_user.id else proposal.proposer
        
        def get_product_public_response(product_obj):
            thumb_url = None
            if product_obj.images:
                sorted_images = sorted(product_obj.images, key=lambda img: img.upload_order or float('inf'))
                if sorted_images:
                    thumb_url = sorted_images[0].image_url
            return ProductPublicResponse(
                id=product_obj.id, 
                title=product_obj.title, 
                description=product_obj.description, 
                thumbnail_image_url=thumb_url
            )

        offered_prod_response = get_product_public_response(proposal.offered_product)
        requested_prod_response = get_product_public_response(proposal.requested_product)

        unread_count = sum(1 for msg in proposal.messages if msg.sender_id != current_user.id and not msg.is_read)
        
        sorted_messages = sorted(proposal.messages, key=lambda m: m.timestamp)
        if sorted_messages:
            last_message_obj = sorted_messages[-1]
            last_message_response = MessageResponse.from_orm(last_message_obj)
        else:
            last_message_response = MessageResponse(
                proposal_id=proposal.id,
                id=0, sender_id=proposal.proposer_user_id,
                text="Propuesta iniciada...",
                timestamp=proposal.created_at, is_read=True
            )

        exchange_details = ExchangeDetailsResponse(
            id=proposal.id, 
            offer=offered_prod_response, 
            request=requested_prod_response,
            created_at=proposal.created_at,
            status=proposal.status,
            proposer_user_id=proposal.proposer_user_id
        )
        
        # --- INICIO DE LA CORRECCIÓN ---
        # Cargar explícitamente las valoraciones recibidas por el otro usuario
        db.refresh(other_user_obj, ['ratings_received'])

        other_user_rating_count = len(other_user_obj.ratings_received)
        if other_user_rating_count > 0:
            other_user_rating_score = sum(r.score for r in other_user_obj.ratings_received) / other_user_rating_count
        else:
            # Aseguramos que la variable siempre exista
            other_user_rating_score = 0.0
        
        user_public_response = UserPublicResponse(
            id=other_user_obj.id,
            full_name=other_user_obj.full_name,
            email=other_user_obj.email,
            avatar=other_user_obj.profile_picture,
            bio=other_user_obj.bio, 
            ubicacion=other_user_obj.ubicacion,
            interests=[CategoryResponse.from_orm(interest) for interest in other_user_obj.interests],
            # Añadimos los campos calculados
            rating_score=other_user_rating_score,
            rating_count=other_user_rating_count
        )
        # --- FIN DE LA CORRECCIÓN ---

        all_messages_response = [MessageResponse.from_orm(msg) for msg in sorted_messages]
        conversations_data.append(ConversationResponse(
            exchange=exchange_details,
            user=user_public_response,
            messages=all_messages_response,
            last_message=last_message_response,
            unread_count=unread_count
        ))
        
    return conversations_data

class MessageCreate(BaseModel):
    proposal_id: int
    text: str
    receiver_id: int #

from datetime import datetime
from sqlalchemy.orm import joinedload
from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse

# ... (otros imports y código de tu archivo)

@app.post("/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    # Accept FormData: proposal_id, optional text, optional image
    proposal_id: int = Form(...),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Validate that at least text or an image is provided
    if not text and not image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Se requiere un texto o una imagen para enviar el mensaje."
        )

    # 1. Fetch the proposal (same as before)
    proposal = db.query(Proposal).options(
        joinedload(Proposal.owner_of_requested_product),
        joinedload(Proposal.proposer)
    ).filter(Proposal.id == proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")

    # 2. Check permissions (same as before)
    is_participant = current_user.id in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]
    if not is_participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para enviar mensajes en esta propuesta.")

    # 3. Process the image if it exists
    saved_image_url: Optional[str] = None
    if image:
        allowed_content_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if image.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de archivo no permitido ({image.content_type}). Solo se aceptan JPEG, PNG, GIF, WEBP."
            )
        try:
            # Use your existing function to save the file
            saved_image_url = await save_upload_file(image)
        except Exception as e:
            print(f"Error al guardar imagen del chat: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No se pudo guardar la imagen.")

    # 4. Create and save the message to the database
    new_message = Message(
        proposal_id=proposal_id,
        sender_id=current_user.id,
        text=text, # Save text (can be None)
        image_url=saved_image_url # Save image URL (can be None)
    )
    db.add(new_message)

    # 5. Update proposal timestamp and reset deletion flags (same as before)
    proposal.updated_at = datetime.utcnow()
    proposal.deleted_by_proposer = False
    proposal.deleted_by_receiver = False
    db.add(proposal)

    # Commit changes early to get the message ID and timestamp
    try:
        db.commit()
        db.refresh(new_message)
    except Exception as e:
        db.rollback()
        print(f"Error al guardar mensaje en BD: {e}")
        raise HTTPException(status_code=500, detail="Error al guardar el mensaje.")

    # 6. Send the real-time message via WebSocket
    recipient_id = (
        proposal.proposer_user_id
        if proposal.owner_of_requested_product_id == current_user.id
        else proposal.owner_of_requested_product_id
    )

    # Construct the payload using the updated MessageResponse Pydantic model
    # (Ensure MessageResponse includes `image_url: Optional[str] = None`)
    message_response_data = MessageResponse.from_orm(new_message).model_dump()
    message_for_ws = {
        "type": "new_message",
        "data": message_response_data # This now includes image_url if present
    }

    # Send the message, using json.dumps with default=str for datetime handling
    await manager.send_personal_message(json.dumps(message_for_ws, default=str), recipient_id)

    # 7. Return the created message as the HTTP response
    return new_message # FastAPI uses MessageResponse to serialize this

@app.get("/users/me/blocked", response_model=List[UserPublicResponse])
async def get_my_blocked_users(
    current_user: User = Depends(get_current_user)
):
    # La relación 'blocked_users' que definimos en el modelo User nos da la lista directamente.
    return current_user.blocked_users

@app.delete("/users/{user_id_to_unblock}/block", status_code=status.HTTP_204_NO_CONTENT)
async def unblock_user(
    user_id_to_unblock: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    user_to_unblock = db.query(User).filter(User.id == user_id_to_unblock).first()

    if not user_to_unblock:
        # No hacemos nada si el usuario no existe, la acción es idempotente.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # Verificamos si el usuario está en la lista de bloqueados antes de intentar quitarlo
    if user_to_unblock in current_user.blocked_users:
        current_user.blocked_users.remove(user_to_unblock)
        db.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put("/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    title: str = Form(...),
    description: str = Form(...),
    condition: str = Form(...),
    category_name: str = Form(...),
    photo_order_ids: str = Form(''),
    deleted_photo_ids: str = Form(''),
    exchange_interest_ids: str = Form(''),
    new_photos: List[UploadFile] = File([])
):
    db_product = db.query(Product).options(
        joinedload(Product.images),
        joinedload(Product.exchange_interests)
    ).filter(Product.id == product_id, Product.user_id == current_user.id).first()
    
    if not db_product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Producto no encontrado o no te pertenece.")

    # 1. Actualizar campos de texto
    db_product.title = title
    db_product.description = description
    db_product.condition = condition

    category = db.query(Category).filter(Category.name == category_name).first()
    if not category:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Categoría no válida.")
    db_product.category_id = category.id

    # 2. Manejar imágenes eliminadas
    if deleted_photo_ids:
        ids_to_delete = {int(id_str) for id_str in deleted_photo_ids.split(',') if id_str.isdigit()}
        images_to_delete = db.query(ProductImage).filter(ProductImage.id.in_(ids_to_delete), ProductImage.product_id == product_id).all()
        for img in images_to_delete:
            # Opcional: eliminar el archivo físico del servidor
            # if os.path.exists(img.image_url.lstrip('/')):
            #     os.remove(img.image_url.lstrip('/'))
            db.delete(img)
        db.flush() # Aplica las eliminaciones a la sesión

    # 3. Reordenar fotos existentes
    if photo_order_ids:
        ordered_ids = [int(id_str) for id_str in photo_order_ids.split(',') if id_str.isdigit()]
        for i, photo_id in enumerate(ordered_ids):
            db.query(ProductImage).filter(ProductImage.id == photo_id, ProductImage.product_id == product_id).update({"upload_order": i + 1, "is_thumbnail": i == 0})

    # 4. Subir nuevas fotos
    # Contamos las fotos que quedarán después de eliminar y antes de añadir nuevas
    current_photo_count = db.query(ProductImage).filter(ProductImage.product_id == product_id).count()
    max_order = db.query(func.max(ProductImage.upload_order)).filter(ProductImage.product_id == product_id).scalar() or 0

    for i, photo in enumerate(new_photos):
        if current_photo_count + i >= 4: # Límite de 4 fotos
            break 
        image_url = await save_upload_file(photo)
        new_img = ProductImage(
            product_id=product_id,
            image_url=image_url,
            is_thumbnail=False, # La portada se define en el reordenamiento
            upload_order=max_order + i + 1
        )
        db.add(new_img)
    
    # 5. Actualizar intereses de intercambio
    if exchange_interest_ids:
        interest_ids_list = {int(id_str) for id_str in exchange_interest_ids.split(',') if id_str.isdigit()}
        interests_to_add = db.query(Category).filter(Category.id.in_(interest_ids_list)).all()
        db_product.exchange_interests = interests_to_add
    else:
        db_product.exchange_interests = []

    db_product.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_product)
    
    # 6. Construir la respuesta final
    sorted_images = sorted(db_product.images, key=lambda img: img.upload_order or float('inf'))
    thumbnail_url = sorted_images[0].image_url if sorted_images else None

    return ProductResponse(
        id=db_product.id,
        user_id=db_product.user_id,
        category_id=db_product.category_id,
        title=db_product.title,
        description=db_product.description,
        current_value_estimate=db_product.current_value_estimate,
        condition=db_product.condition,
        status=db_product.status,
        preffered_exchange_items=db_product.preffered_exchange_items,
        location=db_product.location,
        is_active=db_product.is_active,
        views_count=db_product.views_count,
        created_at=db_product.created_at,
        updated_at=db_product.updated_at,
        category_name=db_product.category_obj.name,
        thumbnail_image_url=thumbnail_url,
        images=[ProductImageResponse.from_orm(img) for img in sorted_images],
        exchange_interests=[interest.name for interest in db_product.exchange_interests],
        user_username=db_product.owner.full_name
    )


@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_product = db.query(Product).filter(Product.id == product_id, Product.user_id == current_user.id).first()
    if not db_product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Producto no encontrado o no te pertenece.")
    db.delete(db_product)
    db.commit()
    return None

class MessageReadStatusUpdate(BaseModel):
    message_ids: List[int]
    is_read: bool

@app.get("/api/aboutus")
async def get_about_us_data():
    """
    Devuelve los datos de la página "Nosotros" para el público general.
    """
    return about_us_data

@app.get("/api/admin/aboutus")
async def admin_get_about_us_data(
    admin: User = Depends(get_current_admin_user) 
):
    """
    Devuelve los datos de "Nosotros" para el panel de administración.
    """
    return about_us_data    

@app.patch("/messages/read_status", status_code=status.HTTP_200_OK)
async def update_message_read_status(
    read_status_update: MessageReadStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    messages_updated_count = 0
    for msg_id in read_status_update.message_ids:
        message = db.query(Message).join(Proposal).filter(
            Message.id == msg_id,
            Message.sender_id != current_user.id,
            (Proposal.proposer_user_id == current_user.id) | (Proposal.owner_of_requested_product_id == current_user.id)
        ).first()
        if message:
            message.is_read = read_status_update.is_read
            db.add(message)
            messages_updated_count += 1
    db.commit()
    return {"message": f"{messages_updated_count} mensajes actualizados.", "updated_count": messages_updated_count}

from fastapi.staticfiles import StaticFiles

@app.post("/payment/create_preference", status_code=status.HTTP_201_CREATED)
async def create_preference(
    purchase_request: CreditPurchaseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Crea una preferencia de pago y registra la transacción inicial como pendiente.
    """
    CREDIT_PACKAGES = {
        2: 1.00,
        5: 2.00,
        10: 5.00
    }
    
    expected_price = CREDIT_PACKAGES.get(purchase_request.quantity)
    if not expected_price or expected_price != purchase_request.unit_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La cantidad o el precio de la compra no son válidos."
        )
        
    try:
        external_ref = f"user_{current_user.id}_credits_{purchase_request.quantity}_{uuid.uuid4()}"

        preference_data = {
            "items": [
                {
                    "title": purchase_request.title,
                    "quantity": 1,
                    "currency_id": "PEN",
                    "unit_price": expected_price
                }
            ],
            "payer": {
                "name": current_user.full_name,
                "email": current_user.email,
                "identification": { "type": "DNI", "number": current_user.dni }
            },
            "back_urls": {
                "success": "https://kambiape.com/payment-success",
                "failure": "https://kambiape.com/payment-failure",
                "pending": "https://kambiape.com/payment-pending"
            },
            # LA LÍNEA DE AUTO_RETURN FUE ELIMINADA
            "external_reference": external_ref,
            "notification_url": "https://kambiape.com/api/webhooks/mercadopago"
        }

        # --- INICIO DE LA CORRECCIÓN ---
        
        # 6. Usa el SDK para crear la preferencia
        preference_response = sdk.preference().create(preference_data)
        
        # 7. VERIFICA LA RESPUESTA ANTES DE USARLA
        # Si la respuesta no tiene un status 201 (creado), algo salió mal.
        if preference_response.get("status") != 201:
            # Imprimimos la respuesta completa para ver el error real
            print("--- ERROR DE MERCADO PAGO ---")
            print(preference_response)
            print("-----------------------------")
            # Lanzamos un error claro
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error al crear la preferencia en Mercado Pago: {preference_response.get('response', {}).get('message', 'Error desconocido')}"
            )

        preference = preference_response["response"]
        
        # --- FIN DE LA CORRECCIÓN ---

        # Crea el registro en tu base de datos
        new_transaction = Transaction(
            user_id=current_user.id,
            mp_preference_id=preference["id"],
            external_reference=external_ref,
            status='pending',
            description=purchase_request.title,
            amount=expected_price,
            currency_id='PEN'
        )
        db.add(new_transaction)
        db.commit()
        
        return {"preference_id": preference["id"], "init_point": preference["init_point"]}

    except Exception as e:
        db.rollback()
        # Si el error viene del HTTPException que lanzamos, lo pasamos directamente
        if isinstance(e, HTTPException):
            raise e
        # Si es otro tipo de error, mantenemos el mensaje genérico
        print(f"Error inesperado al crear preferencia: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {e}"
        )

@app.post("/webhooks/mercadopago", status_code=status.HTTP_200_OK)
async def handle_mercadopago_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Recibe, procesa y actúa sobre las notificaciones de pago de Mercado Pago.
    """
    body = await request.json()
    print("--- WEBHOOK RECIBIDO ---")
    print(body)
    
    # Mercado Pago a veces notifica sobre 'merchant_order' y otras sobre 'payment'
    topic = body.get("topic") or body.get("type")
    
    if topic == "payment" or "payment" in body.get("resource", ""):
        payment_id = body.get("data", {}).get("id")
        if not payment_id:
            # A veces el ID viene en otra parte, seamos flexibles
            resource_url = body.get("resource")
            if resource_url:
                payment_id = resource_url.split('/')[-1]

        if not payment_id:
            print("Webhook: No se pudo encontrar el ID del pago en la notificación.")
            return Response(status_code=status.HTTP_200_OK)

        try:
            print(f"Procesando payment_id: {payment_id}")
            # 1. Obtener la información completa del pago desde Mercado Pago
            payment_info_response = sdk.payment().get(payment_id)
            payment_info = payment_info_response.get("response")

            if not payment_info:
                print(f"Webhook: No se pudo obtener información para el payment_id {payment_id}")
                return Response(status_code=status.HTTP_200_OK)
            
            # 2. Buscar la transacción en nuestra base de datos por external_reference
            external_ref = payment_info.get("external_reference")
            db_transaction = db.query(Transaction).filter(Transaction.external_reference == external_ref).first()

            if not db_transaction:
                print(f"Webhook: No se encontró transacción para external_reference {external_ref}")
                return Response(status_code=status.HTTP_200_OK)

            # 3. Actualizar nuestra transacción solo si el estado ha cambiado
            if db_transaction.status == payment_info["status"]:
                print(f"Transacción {db_transaction.id} ya está en estado '{db_transaction.status}'. No se necesita actualización.")
                return Response(status_code=status.HTTP_200_OK)

            db_transaction.status = payment_info["status"]
            db_transaction.mp_payment_id = payment_id
            db_transaction.updated_at = datetime.utcnow()
            
            # 4. Lógica de negocio: Si el pago fue aprobado, ¡damos los créditos!
            if db_transaction.status == 'approved':
                user_to_credit = db.query(User).filter(User.id == db_transaction.user_id).first()
                if user_to_credit:
                    try:
                        parts = external_ref.split('_')
                        credits_to_add = int(parts[3])
                        user_to_credit.credits += credits_to_add
                        db.add(user_to_credit)
                        print(f"¡ÉXITO! Se añadieron {credits_to_add} créditos al usuario {user_to_credit.id}. Nuevo saldo: {user_to_credit.credits}")
                    except (IndexError, ValueError) as e:
                        print(f"Error al parsear créditos desde external_reference '{external_ref}': {e}")
            
            db.commit()
            print(f"Transacción {db_transaction.id} actualizada a estado '{db_transaction.status}'.")

        except Exception as e:
            db.rollback()
            print(f"Error CRÍTICO procesando webhook de Mercado Pago: {e}")
            # Aún así devolvemos 200 para que MP no siga reintentando un webhook que falla
            return Response(status_code=status.HTTP_200_OK)
            
    return Response(status_code=status.HTTP_200_OK)

@app.post("/payment/process_payment", status_code=status.HTTP_201_CREATED)
async def process_payment(
    payment_data: PaymentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Procesa un pago directamente usando la Checkout API de Mercado Pago.
    """
    try:
        # 1. Prepara el cuerpo de la solicitud para la API de Mercado Pago
        external_ref = f"user_{current_user.id}_credits_api_{uuid.uuid4()}"
        payment_request_body = {
            "transaction_amount": payment_data.transaction_amount,
            "token": payment_data.token,
            "description": payment_data.description,
            "installments": payment_data.installments,
            "payment_method_id": payment_data.payment_method_id,
            "issuer_id": payment_data.issuer_id,
            "payer": payment_data.payer,
            "external_reference": external_ref,
            "notification_url": "https://kambiape.com/api/webhooks/mercadopago" # Asegúrate que tu URL de webhook sea la correcta
        }

        # 2. Usa el SDK para crear el pago (¡esta es la llamada clave!)
        payment_response = sdk.payment().create(payment_request_body)

        if payment_response.get("status") != 201:
            print("--- ERROR DE MERCADO PAGO ---")
            print(payment_response)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=payment_response.get("response", {}).get("message", "Error al procesar el pago.")
            )

        payment_result = payment_response["response"]
        payment_status = payment_result.get("status")
        payment_id_mp = str(payment_result.get("id"))

        # 3. Guarda la transacción en tu base de datos
        new_transaction = Transaction(
            user_id=current_user.id,
            mp_payment_id=payment_id_mp,
            external_reference=external_ref,
            status=payment_status,
            description=payment_data.description,
            amount=payment_data.transaction_amount,
            currency_id='PEN',
            payment_method_id=payment_data.payment_method_id
        )
        db.add(new_transaction)
        
        # 4. Si el pago es aprobado, acredita los créditos INMEDIATAMENTE
        if payment_status == 'approved':
            try:
                # Extraemos la cantidad de créditos desde la descripción
                parts = payment_data.description.split(' ')
                credits_to_add = int(parts[2]) # Asumiendo formato "Compra de 10 Créditos..."
                current_user.credits += credits_to_add
                db.add(current_user)
                print(f"¡ÉXITO! Se añadieron {credits_to_add} créditos al usuario {current_user.id}.")
            except (IndexError, ValueError) as e:
                print(f"ADVERTENCIA: Pago {payment_id_mp} aprobado, pero no se pudieron acreditar créditos automáticamente: {e}")
        
        db.commit()

        # 5. Devuelve el resultado al frontend para que muestre el mensaje final
        return {
            "status": payment_status,
            "detail": payment_result.get("status_detail", "Pago procesado."),
            "id": payment_id_mp
        }

    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        print(f"Error inesperado al procesar el pago: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {e}"
        )
    
@app.get("/admin/chart-data", response_model=ChartDataResponse)
async def get_admin_chart_data(
    metric: Literal['users', 'products', 'proposals', 'transactions'] = Query(...),
    period: Literal['daily', 'weekly', 'monthly', 'yearly'] = Query('monthly'),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Obtiene datos agregados listos para un gráfico, basados en una métrica
    y un período de tiempo.
    """
    
    # 1. Mapear el 'period' a un formato de truncamiento de fecha de SQL
    # (Usamos 'to_char' para formatear la etiqueta, esto es para PostgreSQL.
    # Si usas MySQL, necesitarías cambiarlo por DATE_FORMAT)
    period_format_map = {
        'daily': ('day', 'YYYY-MM-DD'),
        'weekly': ('week', 'YYYY-WW'),
        'monthly': ('month', 'YYYY-MM'),
        'yearly': ('year', 'YYYY'),
    }
    
    if period not in period_format_map:
        raise HTTPException(status_code=400, detail="Período no válido")

    trunc_period, date_format = period_format_map[period]

    # 2. Mapear la 'metric' al modelo y columna de fecha correctos
    metric_config = {
        'users': {
            "model": User,
            "date_col": User.created_at,
            "filter": None,
            "label": "Nuevos Usuarios"
        },
        'products': {
            "model": Product,
            "date_col": Product.created_at,
            "filter": None,
            "label": "Nuevas Publicaciones"
        },
        'proposals': {
            "model": Proposal,
            "date_col": Proposal.updated_at, # Usamos updated_at para capturar cuándo se completó
            "filter": (Proposal.status == 'completed'),
            "label": "Propuestas Completadas"
        },
        'transactions': {
            "model": Transaction,
            "date_col": Transaction.updated_at, # Usamos updated_at para capturar cuándo se aprobó
            "filter": (Transaction.status == 'approved'),
            "label": "Ventas Aprobadas"
        },
    }

    if metric not in metric_config:
        raise HTTPException(status_code=400, detail="Métrica no válida")
        
    config = metric_config[metric]
    model = config["model"]
    date_col = config["date_col"]

    # 3. Construir la consulta
    try:
        # Truncamos la fecha y la formateamos como texto para la etiqueta
        date_label = func.to_char(
            func.date_trunc(trunc_period, date_col),
            date_format
        ).label('label')
        
        query = db.query(
            date_label,
            func.count(model.id).label('count')
        )

        # Aplicar filtros si existen (para propuestas 'completed' o transacciones 'approved')
        if config["filter"] is not None:
            query = query.filter(config["filter"])
            
        # Agrupar y ordenar por la etiqueta de fecha
        results = query.group_by(date_label).order_by(date_label.asc()).all()

        # 4. Formatear la respuesta para Chart.js
        labels = [r.label for r in results]
        data = [r.count for r in results]
        
        return ChartDataResponse(
            labels=labels,
            datasets=[
                {
                    "label": config["label"],
                    "data": data,
                    "backgroundColor": 'rgba(79, 70, 229, 0.2)', # Color Indigo
                    "borderColor": 'rgba(79, 70, 229, 1)',
                    "borderWidth": 2,
                    "tension": 0.1,
                    "fill": True,
                }
            ]
        )

    except Exception as e:
        print(f"Error al generar datos del gráfico: {e}")
        # Captura errores comunes de sintaxis de BD (ej. si no es PostgreSQL)
        if "DATE_FORMAT" in str(e) or "to_char" in str(e):
             raise HTTPException(
                status_code=500,
                detail="Error de base de datos. Es posible que la función de formato de fecha (to_char/DATE_FORMAT) no sea compatible con tu dialecto de SQL."
            )
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar los datos del gráfico."
        )
    
@app.delete("/admin/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_by_admin(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Elimina permanentemente a un usuario. (Acción de Admin)
    """
    if user_id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El administrador no se puede eliminar a sí mismo.")
        
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
    
    db.delete(db_user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put("/admin/users/{user_id}/credits", response_model=UserResponse)
async def update_user_credits(
    user_id: int,
    credit_data: AdminCreditUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Establece el balance de créditos de un usuario. (Acción de Admin)
    """
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
        
    db_user.credits = credit_data.credits
    db.commit()
    db.refresh(db_user)
    
    # Devuelve el usuario actualizado completo (necesario para UserResponse)
    return await get_user_profile(user_id, db, db_user)

@app.put("/admin/users/{user_id}/status", response_model=UserResponse)
async def update_user_status(
    user_id: int,
    status_data: AdminStatusUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Activa o desactiva (banea) a un usuario. (Acción de Admin)
    """
    if user_id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El administrador no se puede desactivar a sí mismo.")
        
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")
        
    db_user.is_active = status_data.is_active
    db.commit()
    db.refresh(db_user)
    
    return await get_user_profile(user_id, db, db_user)

@app.get("/admin/users/{user_id}/proposals", response_model=List[ProposalResponse])
async def get_user_proposals_history(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """
    Obtiene el historial de todas las propuestas (intercambios) de un usuario. (Acción de Admin)
    """
    # Carga todas las propuestas donde el usuario es el proponente O el receptor
    proposals = db.query(Proposal).options(
        joinedload(Proposal.offered_product),
        joinedload(Proposal.requested_product),
        joinedload(Proposal.proposer),
        joinedload(Proposal.owner_of_requested_product)
    ).filter(
        or_(
            Proposal.proposer_user_id == user_id,
            Proposal.owner_of_requested_product_id == user_id
        )
    ).order_by(Proposal.updated_at.desc()).all()
    
    return proposals
    
@app.get("/api/hero", response_model=HeroData)
async def get_hero_data():
    return hero_section_data

# 2. Endpoint de ADMIN para obtener datos del Hero (para HeroSection-admin.vue)
@app.get("/api/admin/hero", response_model=HeroData)
async def get_admin_hero_data(admin: User = Depends(get_current_admin_user)):
    return hero_section_data

# 3. Endpoint de ADMIN para ACTUALIZAR datos del Hero (para HeroSection-admin.vue)
@app.put("/api/admin/hero", response_model=HeroData)
async def update_hero_data(
    # Textos generales (como antes)
    titleLine1: str = Form(...),
    titleLine2: str = Form(...),
    badgeText: str = Form(...),
    paragraphText: str = Form(...),
    button1Text: str = Form(...),
    button2Text: str = Form(...),

    # Textos de las tarjetas (recibidos con índice)
    card_0_alt: str = Form(...),
    card_0_badge: str = Form(...),
    card_1_alt: str = Form(...),
    card_1_badge: str = Form(...),
    card_2_alt: str = Form(...),
    card_2_badge: str = Form(...),

    # Imágenes opcionales de las tarjetas (recibidas con índice)
    card_0_image: Optional[UploadFile] = File(None),
    card_1_image: Optional[UploadFile] = File(None),
    card_2_image: Optional[UploadFile] = File(None),

    admin: User = Depends(get_current_admin_user) # Asegura que solo admin pueda hacer esto
):
    """
    Actualiza todos los textos y las imágenes (opcionales) del Hero Section,
    incluyendo las 3 tarjetas rotativas.
    """
    global hero_section_data # Necesario para modificar la variable global

    # --- Procesamiento de Imágenes de Tarjetas ---
    # Mantiene las URLs actuales como fallback si no se sube nueva imagen
    new_card_image_urls = [
        hero_section_data["cards"][0].get("imageUrl") if len(hero_section_data.get("cards", [])) > 0 else None,
        hero_section_data["cards"][1].get("imageUrl") if len(hero_section_data.get("cards", [])) > 1 else None,
        hero_section_data["cards"][2].get("imageUrl") if len(hero_section_data.get("cards", [])) > 2 else None
    ]
    # Lista de archivos subidos
    card_images_files = [card_0_image, card_1_image, card_2_image]

    for i, image_file in enumerate(card_images_files):
        if image_file: # Si se subió un archivo para esta tarjeta
            try:
                # --- Opcional: Lógica para borrar imagen antigua ---
                # old_image_path = hero_section_data["cards"][i].get("imageUrl")
                # # Evita borrar imágenes por defecto o rutas inválidas
                # if old_image_path and old_image_path.startswith(f"/{UPLOAD_DIR}/"):
                #     full_old_path = os.path.join(os.getcwd(), old_image_path.lstrip('/'))
                #     if os.path.exists(full_old_path):
                #         try:
                #             os.remove(full_old_path)
                #             print(f"Imagen anterior de tarjeta {i} eliminada: {full_old_path}")
                #         except OSError as e:
                #             print(f"Error al eliminar imagen anterior de tarjeta {i} ({full_old_path}): {e}")
                # --- Fin Lógica Opcional ---

                # Guardar la nueva imagen y obtener su URL web
                new_url = await save_upload_file(image_file)
                new_card_image_urls[i] = new_url # Actualizar la URL para esta tarjeta en nuestra lista temporal
                print(f"Nueva imagen para tarjeta {i} guardada en: {new_url}")

            except Exception as e:
                # Si falla el guardado de una imagen, registrar y devolver error
                print(f"Error crítico al procesar imagen para tarjeta {i}: {e}")
                # Puedes decidir si continuar con las otras o fallar todo
                raise HTTPException(status_code=500, detail=f"Error interno al guardar la imagen para la tarjeta {i+1}.")

    # --- Actualización de Datos Globales ---
    # Actualizar textos generales
    hero_section_data["titleLine1"] = titleLine1
    hero_section_data["titleLine2"] = titleLine2
    hero_section_data["badgeText"] = badgeText
    hero_section_data["paragraphText"] = paragraphText
    hero_section_data["button1Text"] = button1Text
    hero_section_data["button2Text"] = button2Text

    # Actualizar la lista de tarjetas con los nuevos textos y las URLs de imagen procesadas
    hero_section_data["cards"] = [
        # Usamos los IDs fijos 1, 2, 3 o podríamos generarlos si fuera necesario
        {"id": 1, "alt": card_0_alt, "badge": card_0_badge, "imageUrl": new_card_image_urls[0]},
        {"id": 2, "alt": card_1_alt, "badge": card_1_badge, "imageUrl": new_card_image_urls[1]},
        {"id": 3, "alt": card_2_alt, "badge": card_2_badge, "imageUrl": new_card_image_urls[2]},
    ]

    print("Datos completos del Hero actualizados:", hero_section_data) # Log para depuración

    # --- Validación y Respuesta ---
    # Validar que la estructura final coincida con el modelo Pydantic HeroData
    try:
        validated_data = HeroData(**hero_section_data)
        # Devolver el objeto validado (Pydantic lo convierte a JSON)
        return validated_data
    except Exception as e: # Captura errores de validación Pydantic (ej. si falta un campo)
        print(f"Error de validación Pydantic al preparar la respuesta: {e}")
        # En producción, podrías querer devolver un error 500 más genérico
        raise HTTPException(status_code=500, detail=f"Error interno al validar los datos actualizados: {e}")

@app.get("/admin/users/{user_id}/blocked", response_model=List[UserPublicResponse])
async def get_users_blocked_by_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user) # Protegido por admin
):
    """
    Devuelve la lista de usuarios que el usuario con user_id ha bloqueado.
    """
    user = db.query(User).options(
        joinedload(User.blocked_users) # Carga eficiente de la relación
    ).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # La relación 'blocked_users' ya contiene la lista de objetos User
    # Pydantic se encargará de serializarlos usando UserPublicResponse
    return user.blocked_users

@app.get("/admin/users/{user_id}/blocked-by", response_model=List[UserPublicResponse])
async def get_users_who_blocked_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin_user) # Protegido por admin
):
    """
    Devuelve la lista de usuarios que han bloqueado al usuario con user_id.
    """
    user = db.query(User).options(
        joinedload(User.blocked_by_users) # Carga eficiente de la relación (backref)
    ).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # La relación 'blocked_by_users' (definida por backref) contiene la lista
    return user.blocked_by_users

@app.put("/api/admin/aboutus")
async def admin_update_about_us_data(
    # --- Dependencia de Admin ---
    admin: User = Depends(get_current_admin_user),
    
    # --- 1. Hero Section ---
    hero_badge: str = Form(...),
    hero_title: str = Form(...),
    hero_paragraph: str = Form(...),
    hero_btn1: str = Form(...),
    hero_btn2: str = Form(...),
    
    # --- 2. Hero Cards (Textos) ---
    heroCard_0_alt: str = Form(...),
    heroCard_0_caption: str = Form(...),
    heroCard_0_title: str = Form(...),
    heroCard_1_alt: str = Form(...),
    heroCard_1_caption: str = Form(...),
    heroCard_1_title: str = Form(...),
    heroCard_2_alt: str = Form(...),
    heroCard_2_caption: str = Form(...),
    heroCard_2_title: str = Form(...),
    
    # --- Hero Cards (Imágenes) ---
    heroCard_0_image: Optional[UploadFile] = File(None),
    heroCard_1_image: Optional[UploadFile] = File(None),
    heroCard_2_image: Optional[UploadFile] = File(None),

    # --- 3. About Section ---
    about_title: str = Form(...),
    about_paragraph: str = Form(...),
    about_quote: str = Form(...),
    
    # --- 4. Tabs (Textos) ---
    tab_0_paragraph: str = Form(...),
    tab_0_quote: str = Form(...),
    tab_0_list: str = Form(...),
    tab_0_footer: str = Form(...),
    tab_0_alt: str = Form(...),
    tab_0_caption: str = Form(...),
    
    tab_1_paragraph: str = Form(...),
    tab_1_quote: str = Form(...),
    tab_1_list: str = Form(...),
    tab_1_footer: str = Form(...),
    tab_1_alt: str = Form(...),
    tab_1_caption: str = Form(...),
    
    tab_2_paragraph: str = Form(...),
    tab_2_quote: str = Form(...),
    tab_2_list: str = Form(...),
    tab_2_footer: str = Form(...),
    tab_2_alt: str = Form(...),
    tab_2_caption: str = Form(...),
    
    # --- Tabs (Imágenes) ---
    tab_0_image: Optional[UploadFile] = File(None),
    tab_1_image: Optional[UploadFile] = File(None),
    tab_2_image: Optional[UploadFile] = File(None),
    
    # --- 5. Community Section ---
    community_title: str = Form(...),
    community_paragraph: str = Form(...),
    community_btnText: str = Form(...),
    community_link: str = Form(...),
    
    # --- 6. Social Section ---
    social_insta: str = Form(...),
    social_tiktok: str = Form(...),
    social_facebook: str = Form(...)
):
    """
    Actualiza el contenido de la página "Nosotros" con datos del formulario admin.
    """
    global about_us_data
    
    try:
        # --- 1. Actualizar Hero ---
        about_us_data["hero"] = {
            "badge": hero_badge, "title": hero_title, "paragraph": hero_paragraph,
            "btn1": hero_btn1, "btn2": hero_btn2
        }
        
        # --- 2. Actualizar Hero Cards ---
        hero_cards_data = [
            {"alt": heroCard_0_alt, "caption": heroCard_0_caption, "title": heroCard_0_title},
            {"alt": heroCard_1_alt, "caption": heroCard_1_caption, "title": heroCard_1_title},
            {"alt": heroCard_2_alt, "caption": heroCard_2_caption, "title": heroCard_2_title}
        ]
        hero_card_images = [heroCard_0_image, heroCard_1_image, heroCard_2_image]
        
        for i, (data, img) in enumerate(zip(hero_cards_data, hero_card_images)):
            about_us_data["heroCards"][i].update(data)
            if img:
                img_url = await save_upload_file(img)
                if img_url:
                    about_us_data["heroCards"][i]["imageUrl"] = img_url

        # --- 3. Actualizar About ---
        about_us_data["about"] = {
            "title": about_title, "paragraph": about_paragraph, "quote": about_quote
        }
        
        # --- 4. Actualizar Tabs ---
        tabs_data = [
            {"paragraph": tab_0_paragraph, "quote": tab_0_quote, "list": tab_0_list, "footer": tab_0_footer, "alt": tab_0_alt, "caption": tab_0_caption},
            {"paragraph": tab_1_paragraph, "quote": tab_1_quote, "list": tab_1_list, "footer": tab_1_footer, "alt": tab_1_alt, "caption": tab_1_caption},
            {"paragraph": tab_2_paragraph, "quote": tab_2_quote, "list": tab_2_list, "footer": tab_2_footer, "alt": tab_2_alt, "caption": tab_2_caption}
        ]
        tab_images = [tab_0_image, tab_1_image, tab_2_image]
        
        for i, (data, img) in enumerate(zip(tabs_data, tab_images)):
            about_us_data["tabs"][i].update(data)
            if img:
                img_url = await save_upload_file(img)
                if img_url:
                    about_us_data["tabs"][i]["imageUrl"] = img_url
        
        # --- 5. Actualizar Community ---
        about_us_data["community"] = {
            "title": community_title, "paragraph": community_paragraph,
            "btnText": community_btnText, "link": community_link
        }
        
        # --- 6. Actualizar Social ---
        about_us_data["social"] = {
            "insta": social_insta, "tiktok": social_tiktok, "facebook": social_facebook
        }
        
        # Devolver todos los datos actualizados
        return about_us_data

    except Exception as e:
        print(f"Error al actualizar 'Nosotros': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ocurrió un error al guardar los datos.")

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")