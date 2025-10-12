from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Date, Text, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Literal, Set # <-- LÍNEA CORREGIDA Y UNIFICADA
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

#Importaciones para la validacion de dni con PeruDevs
import requests
import re

# --- Configuración de la base de datos ---
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("La variable de entorno DATABASE_URL no está configurada. Crea un archivo .env con DATABASE_URL=postgresql://usuario:contraseña@host:puerto/nombre_bd")

# Carga las variables del API de Perudevs para usarlas globalmente
PERUDEVS_API_KEY = os.getenv("PERUDEVS_DNI_KEY")
PERUDEVS_DNI_URL = os.getenv("PERUDEVS_DNI_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)
    occupation = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    dni = Column(String(12), nullable=True, unique=True)
    credits = Column(Integer, default=10, nullable=False)
    # He notado que usas 'user_interests_table' aquí, asegúrate de que esa sea 
    # la variable correcta donde definiste la tabla de unión. 
    # Si la llamaste 'user_interests' como en mi paso anterior, usa ese nombre.
    interests = relationship("Category", secondary=user_interests_table, back_populates="interested_users", lazy="joined")
    profile_picture = Column(String(500), nullable=True)

    products_owned = relationship("Product", back_populates="owner")
    proposals_made = relationship("Proposal", foreign_keys="[Proposal.proposer_user_id]", back_populates="proposer")
    proposals_received = relationship("Proposal", foreign_keys="[Proposal.owner_of_requested_product_id]", back_populates="owner_of_requested_product")

    blocked_users = relationship("User", 
                                 secondary=user_blocks, # <-- Asegúrate que esta tabla se llame 'user_blocks'
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

class ConnectionManager:
    def __init__(self):
        # user_id -> Set de WebSockets (permite múltiples pestañas)
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        print(f"Usuario {user_id} conectado.")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections and websocket in self.active_connections[user_id]:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        print(f"Usuario {user_id} desconectado.")

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            # Enviamos el mensaje a todas las conexiones activas del usuario
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)

manager = ConnectionManager()

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


# NUEVO: Modelo para los mensajes de chat
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    proposal_id = Column(Integer, ForeignKey("proposals.id"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False) # El usuario que envió el mensaje
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow) # Ajustado a timezone=True
    is_read = Column(Boolean, default=False) # Indica si el receptor ha leído el mensaje

    proposal = relationship("Proposal", back_populates="messages")
    sender_obj = relationship("User", foreign_keys=[sender_id], lazy="joined")


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="KambiaPe API",
    description="API para la gestión de usuarios y transacciones de KambiaPe.",
    version="0.0.1",
)

from fastapi.middleware.cors import CORSMiddleware

# CONFIGURACIÓN DE CORS
origins = [
    "http://localhost:5173",    # La dirección principal de Vite
    "http://127.0.0.1:5173",  # A veces el navegador usa esta IP, es bueno tenerla
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    agreed_terms: bool
    created_at: datetime
    phone: str | None = None
    ubicacion: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    occupation: str | None = None
    bio: str | None = None
    dni: str | None = None
    credits: int
    interests: List[str] = []
    profile_picture: str | None = None

    class Config:
        from_attributes = True

# --- Modelos Pydantic para JWT ---
class Token(BaseModel):
    access_token: str
    token_type: str

class ProposalStatusUpdate(BaseModel):
    status: Literal['accepted', 'rejected', 'cancelled']

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str
    confirm_new_password: str


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


# Esquema Pydantic para UserBasicInfo (información básica del usuario) - Renombrado/Ajustado a UserPublicResponse
class UserPublicResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    bio: Optional[str] = None 
    ubicacion: Optional[str] = None
    interests: List[CategoryResponse] = []
    avatar: Optional[str] = None # Campo para simular el avatar en el frontend
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

# NUEVO: Pydantic model para un mensaje individual
class MessageResponse(BaseModel):
    id: int
    sender_id: int
    text: str
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
    api_data = validate_dni_data(dni=user.dni, full_name=user.full_name) # <-- CAMBIO CLAVE

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
    
    # 6. Preparación de la respuesta
    user_data = {
        "id": db_user.id, "full_name": db_user.full_name, "email": db_user.email,
        "agreed_terms": db_user.agreed_terms, "created_at": db_user.created_at, "phone": db_user.phone,
        "ubicacion": db_user.ubicacion,
        "occupation": db_user.occupation, "bio": db_user.bio, "dni": db_user.dni,
        "profile_picture": db_user.profile_picture, "credits": db_user.credits,
        "interests": [interest.name for interest in db_user.interests]
    }
    return UserResponse(**user_data)

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

@app.put("/proposals/{proposal_id}/status", response_model=ProposalResponse) # <-- ¡CAMBIO AQUÍ!
async def update_proposal_status(
    proposal_id: int,
    status_update: ProposalStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Actualiza el estado de una propuesta.
    - 'accepted' o 'rejected' solo puede ser ejecutado por el dueño del producto solicitado.
    - 'cancelled' solo puede ser ejecutado por el usuario que hizo la propuesta (proposer).
    """
    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")

    new_status = status_update.status
    user_id = current_user.id

    # Lógica de permisos para aceptar o rechazar
    if new_status in ['accepted', 'rejected']:
        # Solo el dueño del producto que se está pidiendo puede aceptar o rechazar
        if user_id != proposal.owner_of_requested_product_id:
            raise HTTPException(status_code=status.HTTP_4303_FORBIDDEN, detail="No tienes permiso para aceptar o rechazar esta propuesta.")
        if proposal.status != 'pending':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No se puede {new_status} una propuesta que no está pendiente.")
    
    # Lógica de permisos para cancelar
    elif new_status == 'cancelled':
        # Solo el que hizo la oferta (proposer) puede cancelarla
        if user_id != proposal.proposer_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No puedes cancelar una propuesta que no hiciste.")
        if proposal.status != 'pending':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se puede cancelar una propuesta que ya fue aceptada o rechazada.")
    
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Estado no válido.")

    # Si pasa los permisos, actualiza el estado
    proposal.status = new_status
    db.commit()
    db.refresh(proposal)
    
    return proposal

@app.get("/profile/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para ver este perfil.")
    
    user = db.query(User).options(joinedload(User.interests)).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    user_data = {
        "id": user.id, "full_name": user.full_name, "email": user.email,
        "agreed_terms": user.agreed_terms, "created_at": user.created_at, "phone": user.phone,
        "ubicacion": user.ubicacion, "date_of_birth": user.date_of_birth, "gender": user.gender,
        "occupation": user.occupation, "bio": user.bio, "dni": user.dni,
        "profile_picture": user.profile_picture, "interests": [interest.name for interest in user.interests],
        "credits": user.credits
    }
    return UserResponse(**user_data)

class UserUpdate(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    ubicacion: str | None = None
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

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, token: str = Query(...)):
    # Obtenemos la sesión de la base de datos
    db: Session = next(get_db())
    try:
        # Verificamos el token para autenticar al usuario
        current_user = get_current_user_from_token(token, db)
        
        # Si el token es inválido o no corresponde al user_id de la URL, rechazamos la conexión
        if not current_user or current_user.id != user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Token inválido o no autorizado.")
            return

        # Si el token es válido, aceptamos y manejamos la conexión
        await manager.connect(websocket, user_id)
        
        try:
            while True:
                # La lógica para recibir y enviar mensajes se mantiene
                data = await websocket.receive_json()
                
                recipient_id = data.get('recipient_id')
                proposal_id = data.get('proposal_id')
                text = data.get('text')

                if recipient_id and proposal_id and text:
                    # Guardamos el nuevo mensaje
                    new_msg = Message(proposal_id=proposal_id, sender_id=user_id, text=text)
                    db.add(new_msg)
                    db.commit()
                    db.refresh(new_msg)

                    # Preparamos el mensaje para enviarlo
                    message_to_send = {
                        "type": "new_message",
                        # Usamos .model_dump() de Pydantic para convertir el objeto a dict
                        "data": MessageResponse.from_orm(new_msg).model_dump()
                    }
                    
                    # Usamos json.dumps para enviar un string, que es más estándar
                    await manager.send_personal_message(json.dumps(message_to_send, default=str), recipient_id)

        except WebSocketDisconnect:
            manager.disconnect(websocket, user_id) # Pasamos ambos argumentos
        except Exception as e:
            print(f"Error inesperado en WebSocket para usuario {user_id}: {e}")
            manager.disconnect(websocket, user_id) # Pasamos ambos argumentos

    finally:
        # Cerramos la sesión de la base de datos al finalizar
        db.close()

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
    ).filter(Product.is_active == True, Product.status == 'available').all()

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
    # --- ✨ 1. VERIFICACIÓN Y DESCUENTO DE CRÉDITOS ---
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

    # --- 2. VALIDACIONES DEL PRODUCTO ---
    if not all([title.strip(), description.strip(), category_name.strip(), condition.strip()]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Los campos de título, descripción, categoría y condición son obligatorios.")
    if not photos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Se requiere al menos una imagen.")
    if len(photos) > 4:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Solo se permiten un máximo de 4 imágenes.")

    # --- 3. LÓGICA DE NEGOCIO (Categorías e Intereses) ---
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

    # --- 4. CREACIÓN DEL PRODUCTO Y GUARDADO DE IMÁGENES ---
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

    # --- 5. ASIGNACIÓN DE INTERESES Y COMMIT FINAL ---
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
        
    # --- 6. CONSTRUCCIÓN DE LA RESPUESTA ---
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
    # --- INICIO DE LA MODIFICACIÓN ---

    # 1. Obtener una lista de IDs de usuarios que el usuario actual ha bloqueado
    #    y también los que lo han bloqueado a él.
    blocked_user_ids = {user.id for user in current_user.blocked_users}
    users_who_blocked_current_user = {user.id for user in current_user.blocked_by_users}
    all_blocked_ids = blocked_user_ids.union(users_who_blocked_current_user)

    # --- FIN DE LA MODIFICACIÓN ---

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
        # --- INICIO DE LA MODIFICACIÓN ---
        # 3. Añadir este filtro para excluir conversaciones con usuarios bloqueados
        Proposal.proposer_user_id.notin_(all_blocked_ids),
        Proposal.owner_of_requested_product_id.notin_(all_blocked_ids)
        # --- FIN DE LA MODIFICACIÓN ---
    ).order_by(Proposal.updated_at.desc()).all()

    # El resto del código de la función sigue exactamente igual...
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
        
        user_public_response = UserPublicResponse(
            id=other_user_obj.id,
            full_name=other_user_obj.full_name,
            email=other_user_obj.email,
            avatar=other_user_obj.profile_picture,
            bio=other_user_obj.bio, 
            ubicacion=other_user_obj.ubicacion,
            interests=other_user_obj.interests
        )

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

from datetime import datetime
from sqlalchemy.orm import joinedload
from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse

# ... (otros imports y código de tu archivo)

@app.post("/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    message_data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1. Validación y obtención de la propuesta
    proposal = db.query(Proposal).options(
        joinedload(Proposal.owner_of_requested_product),
        joinedload(Proposal.proposer)
    ).filter(Proposal.id == message_data.proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")

    # 2. Verificación de permisos
    is_participant = current_user.id in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]
    if not is_participant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para enviar mensajes en esta propuesta.")

    # 3. Creación y guardado del mensaje en la base de datos
    new_message = Message(
        proposal_id=message_data.proposal_id,
        sender_id=current_user.id,
        text=message_data.text
    )
    db.add(new_message)

    # 4. Actualización de la fecha y estado de la propuesta
    proposal.updated_at = datetime.utcnow()
    
    # ===== INICIO DE LA CORRECCIÓN =====
    # Al enviar un nuevo mensaje, "revivimos" la conversación para ambos usuarios,
    # reseteando las banderas de borrado.
    proposal.deleted_by_proposer = False
    proposal.deleted_by_receiver = False
    # ===== FIN DE LA CORRECCIÓN =====

    db.add(proposal)
    db.commit()
    db.refresh(new_message)

    # 5. Envío del mensaje en tiempo real vía WebSocket
    # Determinar quién es el destinatario del mensaje
    recipient_id = (
        proposal.proposer_user_id 
        if proposal.owner_of_requested_product_id == current_user.id 
        else proposal.owner_of_requested_product_id
    )

    # Construir el payload del mensaje para el WebSocket
    # Usamos .isoformat() para que el frontend (JavaScript) pueda interpretar la fecha correctamente
    message_for_ws = {
        "type": "new_message",
        "data": {
            "id": new_message.id,
            "proposal_id": new_message.proposal_id,
            "sender_id": new_message.sender_id,
            "text": new_message.text,
            "timestamp": new_message.timestamp.isoformat(),
            "is_read": new_message.is_read
        }
    }
    
    # Enviar el mensaje al destinatario si está conectado
    await manager.send_personal_message(message_for_ws, recipient_id)

    # 6. Devolver el mensaje creado como respuesta HTTP
    return MessageResponse.from_orm(new_message)
1

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

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")