import bcrypt

# --- ¡PONE TU CONTRASEÑA DE ADMIN AQUI! ---
password_plana = 'ADMINPE!'
# ------------------------------------------

# Codificamos la contraseña a bytes
password_bytes = password_plana.encode('utf-8')

# Generamos el 'salt' (un componente aleatorio)
salt = bcrypt.gensalt()

# Creamos el hash
hash_bytes = bcrypt.hashpw(password_bytes, salt)

# Lo decodificamos para poder copiarlo y pegarlo
hash_string = hash_bytes.decode('utf-8')

print("¡Copia este hash y pégalo en tu comando SQL!")
print(hash_string)