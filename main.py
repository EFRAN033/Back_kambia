from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Date, Text, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, joinedload
import os
from dotenv import load_dotenv
from typing import List, Optional
from passlib.context import CryptContext
from sqlalchemy import Table

# Importaciones para JWT
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
import aiofiles
import uuid

# --- Configuración de la base de datos ---
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("La variable de entorno DATABASE_URL no está configurada. Crea un archivo .env con DATABASE_URL=postgresql://usuario:contraseña@host:puerto/nombre_bd")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

user_interests_table = Table('user_interests', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('category_id', Integer, ForeignKey('categories.id', ondelete="CASCADE"), primary_key=True)
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
    address = Column(String(255), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)
    occupation = Column(String(100), nullable=True)
    bio = Column(Text, nullable=True)
    dni = Column(String(12), nullable=True, unique=True)
    interests = relationship("Category", secondary=user_interests_table, back_populates="interested_users", lazy="joined")



    products_owned = relationship("Product", back_populates="owner")
    proposals_made = relationship("Proposal", foreign_keys="[Proposal.proposer_user_id]", back_populates="proposer")
    proposals_received = relationship("Proposal", foreign_keys="[Proposal.owner_of_requested_product_id]", back_populates="owner_of_requested_product")


class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    interested_users = relationship("User", secondary=user_interests_table, back_populates="interests")


    products_linked = relationship("Product", back_populates="category_obj")


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
    proposals_offered = relationship("Proposal", foreign_keys="[Proposal.offered_product_id]", back_populates="offered_product")
    proposals_requested = relationship("Proposal", foreign_keys="[Proposal.requested_product_id]", back_populates="requested_product")


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

class UserCreate(BaseModel):
    full_name: str
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
    address: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    occupation: str | None = None
    bio: str | None = None
    dni: str | None = None
    interests: List[str] = []

    class Config:
        from_attributes = True

# --- Modelos Pydantic para JWT ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None

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

class ProductResponse(BaseModel): # Original ProductResponse, usado para el feed y productos individuales
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
    
    category_name: str
    thumbnail_image_url: Optional[str] = None 
    images: List[ProductImageResponse] = []

    class Config:
        from_attributes = True

# Esquema Pydantic para UserBasicInfo (información básica del usuario) - Renombrado/Ajustado a UserPublicResponse
class UserPublicResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    avatar: Optional[str] = None # Campo para simular el avatar en el frontend
    class Config:
        from_attributes = True

# Esquema Pydantic para ProductBasicInfo (información básica del producto) - Renombrado/Ajustado a ProductPublicResponse
class ProductPublicResponse(BaseModel): # Un modelo más ligero para productos en el feed/proposals
    id: int
    title: str # Tu Product model usa 'title', no 'name'
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


# NUEVO: Pydantic model para un mensaje individual
class MessageResponse(BaseModel):
    id: int
    sender_id: int
    text: str
    timestamp: datetime
    is_read: bool

    class Config:
        from_attributes = True


# NUEVO: Pydantic model para la información de intercambio dentro de una conversación
class ExchangeDetailsResponse(BaseModel):
    id: int
    offer: ProductPublicResponse # Producto ofrecido
    request: ProductPublicResponse # Producto solicitado
    message: Optional[str] = None # Mensaje inicial de la propuesta (no usado por ahora en el backend, pero útil para frontend)
    date: datetime # created_at de la propuesta
    status: str

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


@app.get("/")
async def root():
    return {"message": "¡Bienvenido a la API de KambiaPe! ✅"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if user.password != user.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Las contraseñas no coinciden."
        )

    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El correo electrónico ya está registrado."
        )

    hashed_password = get_password_hash(user.password)

    db_user = User(
        full_name=user.full_name,
        email=user.email,
        password_hash=hashed_password,
        agreed_terms=user.agreed_terms,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login", response_model=Token)
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_login.email).first()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Credenciales incorrectas: Correo o contraseña inválidos."
        )

    if not verify_password(user_login.password, db_user.password_hash):
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


# main.py

@app.get("/profile/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para ver este perfil.")
    
    # 1. Obtenemos el usuario y sus intereses (objetos) desde la BD
    user = db.query(User).options(joinedload(User.interests)).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    # 2. ✨ LA CORRECCIÓN CLAVE: Construimos un diccionario manualmente
    #    Esto asegura que los datos tengan el formato correcto ANTES de la validación.
    user_data = {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "agreed_terms": user.agreed_terms,
        "created_at": user.created_at,
        "phone": user.phone,
        "address": user.address,
        "date_of_birth": user.date_of_birth,
        "gender": user.gender,
        "occupation": user.occupation,
        "bio": user.bio,
        "dni": user.dni,
        "interests": [interest.name for interest in user.interests] # Convertimos los objetos a strings
    }
    
    # 3. Devolvemos el diccionario. FastAPI lo validará contra UserResponse y ahora sí funcionará.
    return user_data

class UserUpdate(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    address: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    occupation: str | None = None
    bio: str | None = None
    dni: str | None = None
    interest_ids: Optional[List[int]] = None

class CategoryResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


@app.get("/categories", response_model=List[CategoryResponse])
async def get_all_categories(db: Session = Depends(get_db)):
    """Devuelve una lista de todas las categorías disponibles."""
    categories = db.query(Category).order_by(Category.name).all()
    return categories

# main.py

# --- 🔁 REEMPLAZA ESTA FUNCIÓN COMPLETA 🔁 ---
@app.put("/profile/{user_id}", response_model=UserResponse)
async def update_user_profile(user_id: int, user_data: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para actualizar este perfil.")

    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado.")

    update_data = user_data.model_dump(exclude_unset=True)

    # --- LÓGICA PARA ACTUALIZAR INTERESES ---
    if 'interest_ids' in update_data:
        interest_ids = update_data.pop('interest_ids')
        if interest_ids is not None:
            # Busca los objetos de categoría válidos
            interests = db.query(Category).filter(Category.id.in_(interest_ids)).all()
            # Actualiza la relación
            db_user.interests = interests
        else: # Si se envía una lista vacía o null
            db_user.interests = []

    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Recargamos el perfil para incluir los nombres de los intereses en la respuesta
    return await get_user_profile(user_id, db, db_user)

@app.get("/users/{user_id}/products", response_model=List[ProductResponse])
async def get_user_products(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Obtiene todos los productos de un usuario específico,
    incluyendo el nombre de la categoría y la URL de la imagen principal.
    """
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para ver los productos de otros usuarios.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario no encontrado."
        )

    products_db = db.query(Product).options(
        joinedload(Product.category_obj),
        joinedload(Product.images)
    ).filter(Product.user_id == user_id, Product.is_active == True, Product.status == 'available').all()

    response_products = []
    for product in products_db:
        thumbnail_url = None
        if product.images:
            thumbnail_image = next((img for img in product.images if img.is_thumbnail), product.images[0])
            thumbnail_url = thumbnail_image.image_url

        product_data = product.__dict__
        product_data["category_name"] = product.category_obj.name if product.category_obj else None
        product_data["thumbnail_image_url"] = thumbnail_url
        product_data["images"] = [ProductImageResponse.from_orm(img) for img in product.images]
        
        response_products.append(ProductResponse(**product_data))
    
    return response_products

@app.get("/products_feed", response_model=List[ProductResponse])
async def get_all_products_for_feed(db: Session = Depends(get_db)):
    """
    Obtiene todos los productos disponibles para el feed principal,
    incluyendo el nombre de la categoría y la URL de la imagen principal.
    """
    products_db = db.query(Product).options(
        joinedload(Product.category_obj),
        joinedload(Product.images)
    ).filter(Product.is_active == True, Product.status == 'available').all()

    # --- INICIO DE LA CORRECCIÓN ---
    # En lugar de manipular __dict__, creamos la respuesta directamente.
    
    response_products = []
    for product in products_db:
        thumbnail_url = None
        # Busca la imagen thumbnail de forma segura
        if product.images:
            thumbnail_image = next((img for img in product.images if img.is_thumbnail), None)
            if thumbnail_image:
                thumbnail_url = thumbnail_image.image_url
            else:
                # Si no hay thumbnail específico, usa la primera imagen
                thumbnail_url = product.images[0].image_url

        # Construye el objeto de respuesta Pydantic directamente
        # Esto es más seguro y explícito
        response_product = ProductResponse(
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
            images=[ProductImageResponse.from_orm(img) for img in product.images]
        )
        response_products.append(response_product)
        
    return response_products
    # --- FIN DE LA CORRECCIÓN ---

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """Guarda un archivo subido de forma asíncrona y devuelve su URL relativa."""
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        async with aiofiles.open(file_path, "wb") as buffer:
            while content := await upload_file.read(1024 * 1024):
                await buffer.write(content)
    except Exception as e:
        print(f"Error al guardar el archivo {unique_filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudo guardar la imagen: {upload_file.filename}"
        )
    
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Endpoint para publicar un nuevo producto.
    Recibe los detalles del producto y hasta 4 imágenes.
    """
    if not title.strip() or not description.strip() or not category_name.strip() or not condition.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Los campos de título, descripción, categoría y condición son obligatorios."
        )

    if not photos or len(photos) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Se requiere al menos una imagen para el producto."
        )
    if len(photos) > 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # Cambiado de BAD_BAD_REQUEST a BAD_REQUEST
            detail="Solo se permiten un máximo de 4 imágenes por producto."
        )

    category = db.query(Category).filter(Category.name == category_name).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"La categoría '{category_name}' no existe. Por favor, selecciona una categoría válida."
        )

    new_product = Product(
        user_id=current_user.id,
        category_id=category.id,
        title=title,
        description=description,
        current_value_estimate=current_value_estimate,
        condition=condition,
        preffered_exchange_items=preffered_exchange_items,
        location=location,
    )
    db.add(new_product)
    db.commit()
    db.refresh(new_product)

    for i, photo in enumerate(photos):
        try:
            image_url = await save_upload_file(photo)
            is_thumbnail = (i == 0)
            upload_order = i + 1

            db_image = ProductImage(
                product_id=new_product.id,
                image_url=image_url,
                is_thumbnail=is_thumbnail,
                upload_order=upload_order
            )
            db.add(db_image)
        except Exception as e:
            print(f"Error al guardar la imagen {photo.filename}: {e}")
            db.rollback() 
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al subir la imagen {photo.filename}. Por favor, inténtalo de nuevo."
            )
    db.commit()
    db.refresh(new_product)

    thumbnail_url = None
    if new_product.images:
        thumbnail_image = next((img for img in new_product.images if img.is_thumbnail), new_product.images[0]) # Usar new_product.images[0] como fallback
        thumbnail_url = thumbnail_image.image_url

    response_data = new_product.__dict__
    response_data["category_name"] = new_product.category_obj.name if new_product.category_obj else None 
    response_data["thumbnail_image_url"] = thumbnail_url
    response_data["images"] = [ProductImageResponse.from_orm(img) for img in new_product.images]

    return ProductResponse(**response_data)

@app.post("/proposals", response_model=ProposalResponse, status_code=status.HTTP_201_CREATED)
async def create_proposal(
    proposal: ProposalCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Crea una nueva propuesta de intercambio entre dos productos.
    - El `offered_product_id` es el producto que el usuario actual (proposer) ofrece.
    - El `requested_product_id` es el producto que el usuario actual (proposer) solicita.
    """
    # 1. Verificar que el producto ofrecido exista y pertenezca al usuario actual
    offered_product = db.query(Product).filter(
        Product.id == proposal.offered_product_id,
        Product.user_id == current_user.id
    ).first()
    if not offered_product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El producto ofrecido no existe o no te pertenece."
        )
    
    # 2. Verificar que el producto solicitado exista y no pertenezca al usuario actual
    requested_product = db.query(Product).filter(
        Product.id == proposal.requested_product_id
    ).first()
    if not requested_product:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El producto solicitado no existe."
        )

    if requested_product.user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No puedes solicitar un intercambio por tu propio producto."
        )
    
    # 3. Obtener el ID del propietario del producto solicitado
    owner_of_requested_product_id = requested_product.user_id

    # 4. Verificar si ya existe una propuesta pendiente entre estos dos productos y usuarios
    existing_proposal = db.query(Proposal).filter(
        Proposal.offered_product_id == proposal.offered_product_id,
        Proposal.requested_product_id == proposal.requested_product_id,
        Proposal.proposer_user_id == current_user.id,
        Proposal.status == "pending" # Consideramos solo propuestas pendientes
    ).first()

    if existing_proposal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una propuesta pendiente para este intercambio."
        )

    # 5. Crear la nueva propuesta
    new_proposal = Proposal(
        offered_product_id=proposal.offered_product_id,
        requested_product_id=proposal.requested_product_id,
        proposer_user_id=current_user.id,
        owner_of_requested_product_id=owner_of_requested_product_id,
        status="pending"
    )

    db.add(new_proposal)
    db.commit()
    db.refresh(new_proposal)

    # Cargar los objetos relacionados para la respuesta
    db_proposal = db.query(Proposal).options(
        joinedload(Proposal.offered_product),
        joinedload(Proposal.requested_product),
        joinedload(Proposal.proposer),
        joinedload(Proposal.owner_of_requested_product)
    ).filter(Proposal.id == new_proposal.id).first()

    return db_proposal

# REEMPLAZADO: Tu anterior GET /proposals/my fue reemplazado por esta versión más completa
@app.get("/api/v1/proposals/me", response_model=List[ConversationResponse])
async def get_my_proposals(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Obtiene todas las propuestas de intercambio en las que el usuario actual
    está involucrado (como proponente o como dueño del producto solicitado),
    incluyendo los mensajes asociados.
    """
    proposals = db.query(Proposal).options(
        joinedload(Proposal.offered_product).joinedload(Product.images),
        joinedload(Proposal.requested_product).joinedload(Product.images),
        joinedload(Proposal.proposer),
        joinedload(Proposal.owner_of_requested_product),
        joinedload(Proposal.messages).joinedload(Message.sender_obj) # Cargar mensajes y sus remitentes
    ).filter(
        (Proposal.proposer_user_id == current_user.id) |
        (Proposal.owner_of_requested_product_id == current_user.id)
    ).order_by(Proposal.updated_at.desc()).all() # Ordenar por la última actualización

    conversations_data = []
    for proposal in proposals:
        # Determinar quién es el "otro" usuario en la conversación
        other_user = None
        if proposal.proposer_user_id == current_user.id:
            other_user = proposal.owner_of_requested_product
        else:
            other_user = proposal.proposer

        # Preparar los productos para la respuesta, asegurando que tengan thumbnail_image_url
        def get_product_public_response(product_obj):
            thumbnail_url = None
            if product_obj.images:
                thumbnail_image = next((img for img in product_obj.images if img.is_thumbnail), product_obj.images[0])
                thumbnail_url = thumbnail_image.image_url
            return ProductPublicResponse(
                id=product_obj.id,
                title=product_obj.title, # Tu Product model usa 'title'
                description=product_obj.description,
                thumbnail_image_url=thumbnail_url
            )

        offered_prod_response = get_product_public_response(proposal.offered_product)
        requested_prod_response = get_product_public_response(proposal.requested_product)

        # Contar mensajes no leídos
        # Un mensaje se considera no leído si:
        # 1. El usuario actual NO es el remitente (es el receptor)
        # 2. El campo 'is_read' del mensaje es False
        unread_count = sum(
            1 for msg in proposal.messages
            if msg.sender_id != current_user.id and not msg.is_read
        )

        # Obtener el último mensaje
        last_message = None
        if proposal.messages:
            last_message = MessageResponse.from_orm(proposal.messages[-1])
        else:
            # Si no hay mensajes de chat, el "último mensaje" podría ser la propia propuesta inicial
            # Simulamos un MessageResponse para que siempre haya un 'last_message'
            last_message = MessageResponse(
                id=0, # ID dummy, ya que no es un mensaje real de DB
                sender_id=proposal.proposer_user_id, # El proponente como sender
                text=f"Propuesta de {proposal.offered_product.title} por {proposal.requested_product.title}",
                timestamp=proposal.created_at,
                is_read=True # Asumimos que la "propuesta inicial" siempre está "leída" por su propia creación
            )

        # Crear el objeto ExchangeDetailsResponse
        exchange_details = ExchangeDetailsResponse(
            id=proposal.id,
            offer=offered_prod_response,
            request=requested_prod_response,
            message=None, # El mensaje inicial no se guarda directamente en Proposal model. Podrías añadirlo.
            date=proposal.created_at,
            status=proposal.status
        )

        conversations_data.append(ConversationResponse(
            exchange=exchange_details,
            user=UserPublicResponse(
                id=other_user.id,
                full_name=other_user.full_name,
                email=other_user.email,
                avatar=f"https://i.pravatar.cc/150?img={other_user.id % 70}" # Simulación de avatar
            ),
            messages=[MessageResponse.from_orm(msg) for msg in proposal.messages],
            last_message=last_message,
            unread_count=unread_count
        ))
    
    return conversations_data

@app.put("/proposals/{proposal_id}/status")
async def update_proposal_status(
    proposal_id: int,
    status: str, # Puede ser 'accepted' o 'rejected'
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Actualiza el estado de una propuesta (aceptar o rechazar).
    Solo el propietario del producto solicitado puede cambiar el estado.
    """
    proposal = db.query(Proposal).filter(Proposal.id == proposal_id).first()

    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")
    
    # Solo el dueño del producto solicitado puede aceptar/rechazar la propuesta
    if proposal.owner_of_requested_product_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para modificar esta propuesta.")
    
    if status not in ["accepted", "rejected"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Estado inválido. Debe ser 'accepted' o 'rejected'.")
    
    if proposal.status != "pending":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"La propuesta ya no está pendiente. Estado actual: {proposal.status}")

    proposal.status = status
    proposal.updated_at = datetime.utcnow()
    db.add(proposal)
    db.commit()
    db.refresh(proposal)

    return {"message": f"Propuesta {proposal_id} actualizada a estado '{status}' con éxito."}


# NUEVO: Endpoint para enviar mensajes dentro de una propuesta
class MessageCreate(BaseModel):
    proposal_id: int
    text: str

@app.post("/api/v1/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    message_data: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    proposal = db.query(Proposal).filter(Proposal.id == message_data.proposal_id).first()
    if not proposal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Propuesta no encontrada.")

    # Verificar que el usuario actual es parte de la propuesta para enviar un mensaje
    if current_user.id not in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permiso para enviar mensajes a esta propuesta.")
    
    # Opcional: Podrías restringir el envío de mensajes a propuestas que no estén rechazadas/completadas
    # if proposal.status in ["rejected", "completed"]:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No se pueden enviar mensajes a propuestas en estado rechazado o completado.")

    new_message = Message(
        proposal_id=message_data.proposal_id,
        sender_id=current_user.id,
        text=message_data.text,
        timestamp=datetime.utcnow(),
        is_read=False # Por defecto, no leído por el receptor
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    # Actualizar el 'updated_at' de la propuesta para que se refleje en el orden de las conversaciones
    proposal.updated_at = datetime.utcnow()
    db.add(proposal)
    db.commit() # Un segundo commit para la propuesta

    return MessageResponse.from_orm(new_message)

# NUEVO: Endpoint para marcar mensajes como leídos
class MessageReadStatusUpdate(BaseModel):
    message_ids: List[int]
    is_read: bool

@app.patch("/api/v1/messages/read_status", status_code=status.HTTP_200_OK)
async def update_message_read_status(
    read_status_update: MessageReadStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Marca uno o varios mensajes como leídos o no leídos.
    """
    messages_updated_count = 0
    for msg_id in read_status_update.message_ids:
        message = db.query(Message).filter(Message.id == msg_id).first()
        if message:
            # Asegurarse de que el usuario actual es el RECEPTOR del mensaje
            # Es decir, el mensaje no fue enviado por el usuario actual.
            proposal = db.query(Proposal).filter(Proposal.id == message.proposal_id).first()
            if proposal and message.sender_id != current_user.id and \
               current_user.id in [proposal.proposer_user_id, proposal.owner_of_requested_product_id]:
                
                message.is_read = read_status_update.is_read
                db.add(message)
                messages_updated_count += 1
    
    db.commit() # Commit fuera del bucle para eficiencia

    return {"message": f"{messages_updated_count} mensajes actualizados.", "updated_count": messages_updated_count}

from fastapi.staticfiles import StaticFiles

app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")