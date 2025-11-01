save:
	git add .
	git commit -m "update backend"
	git push origin main

# --- Variables de Despliegue (NUEVO) ---
USER = efran18
IMAGE_NAME = kambiape-back
IMAGE = $(USER)/$(IMAGE_NAME):latest

# --- Variables del Servidor (NUEVO) ---
VPS_USER = administrator
VPS_HOST = 83.147.39.44
# Ruta exacta al archivo en tu VPS
COMPOSE_FILE = /home/administrator/docker-compose.yml

# --- Tareas de Despliegue (NUEVO) ---

# 1. Construye la imagen
build:
	docker build -t $(IMAGE) .

# 2. Sube la imagen
push:
	docker push $(IMAGE)

# 3. Actualiza el VPS (NOTA: aquí dice "backend")
update-vps:
	ssh $(VPS_USER)@$(VPS_HOST) "docker-compose -f $(COMPOSE_FILE) pull backend && docker-compose -f $(COMPOSE_FILE) rm --stop -f backend && docker-compose -f $(COMPOSE_FILE) up -d"

# 4. El deploy del backend
deploy: build push update-vps