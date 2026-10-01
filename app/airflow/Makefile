# Makefile - MLOps Airflow Shortcuts
# Uso:
#   make help              - Ver comandos disponibles
#   make up                - Levantar servicios
#   make down              - Detener servicios
#   make dag-basic         - Ejecutar DAG básico
#   make logs-scheduler    - Ver logs del scheduler
#   make flower            - Abrir Flower en navegador
#   make clean             - Limpiar TODO (⚠️ BORRA DATOS)

.PHONY: help up down restart logs logs-scheduler logs-workers \
        dag-basic dag-dynamic dag-retrain dag-list flower \
        db-shell redis-cli status clean rebuild

COMPOSE := docker compose

help:
	@echo "🚀 MLOps Airflow - Comandos disponibles"
	@echo ""
	@echo "INICIAR/DETENER:"
	@echo "  make up              Levantar todos los servicios"
	@echo "  make down            Detener servicios (sin borrar datos)"
	@echo "  make restart         Reiniciar servicios"
	@echo "  make rebuild         Rebuild y reiniciar TODO"
	@echo ""
	@echo "EJECUTAR DAGs:"
	@echo "  make dag-basic       Ejecutar mlops_pipeline_basic"
	@echo "  make dag-dynamic     Ejecutar mlops_dynamic_mapping"
	@echo "  make dag-retrain     Ejecutar mlops_continuous_retraining"
	@echo "  make dag-list        Listar todos los DAGs"
	@echo ""
	@echo "LOGGING:"
	@echo "  make logs            Ver logs de Airflow"
	@echo "  make logs-scheduler  Ver logs del scheduler"
	@echo "  make logs-workers    Ver logs de workers"
	@echo "  make logs-flower     Ver logs de Flower"
	@echo ""
	@echo "MONITOREO:"
	@echo "  make status          Ver estado de servicios"
	@echo "  make flower          Abrir Flower (http://localhost:5555)"
	@echo "  make airflow         Abrir Airflow UI (http://localhost:8080)"
	@echo ""
	@echo "BASE DE DATOS:"
	@echo "  make db-shell        Entrar a shell de PostgreSQL"
	@echo "  make redis-cli       Conectar a Redis CLI"
	@echo ""
	@echo "LIMPIAR:"
	@echo "  make clean           Detener y borrar TODO (⚠️  DATOS PERDIDOS)"
	@echo ""

# ========== INICIAR/DETENER ==========

up:
	@echo "🚀 Levantando Airflow..."
	$(COMPOSE) up -d
	@sleep 5
	@echo "✅ Airflow listo"
	@echo "   UI: http://localhost:8080 (admin/admin)"
	@echo "   Flower: http://localhost:5555"

down:
	@echo "🛑 Deteniendo servicios..."
	$(COMPOSE) down
	@echo "✅ Detenido"

restart: down up
	@echo "✅ Reiniciado"

rebuild:
	@echo "🔨 Reconstruyendo..."
	$(COMPOSE) down
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d
	@sleep 5
	@echo "✅ Reconstruido y levantado"

# ========== EJECUTAR DAGs ==========

dag-basic:
	@echo "▶️ Ejecutando mlops_pipeline_basic..."
	$(COMPOSE) exec -T airflow-scheduler airflow dags trigger mlops_pipeline_basic
	@echo "📊 Accede a http://localhost:8080 para ver ejecución"

dag-dynamic:
	@echo "▶️ Ejecutando mlops_dynamic_mapping..."
	$(COMPOSE) exec -T airflow-scheduler airflow dags trigger mlops_dynamic_mapping
	@echo "📊 Accede a http://localhost:8080 para ver ejecución"

dag-retrain:
	@echo "▶️ Ejecutando mlops_continuous_retraining..."
	$(COMPOSE) exec -T airflow-scheduler airflow dags trigger mlops_continuous_retraining
	@echo "📊 Accede a http://localhost:8080 para ver ejecución"

dag-list:
	@echo "📋 DAGs disponibles:"
	$(COMPOSE) exec -T airflow-scheduler airflow dags list

# ========== LOGGING ==========

logs:
	$(COMPOSE) logs -f

logs-scheduler:
	@echo "📜 Logs del scheduler (Ctrl+C para salir):"
	$(COMPOSE) logs -f airflow-scheduler

logs-workers:
	@echo "📜 Logs de workers (Ctrl+C para salir):"
	@echo "Data worker:"
	$(COMPOSE) logs -f airflow-worker-data

logs-flower:
	@echo "📜 Logs de Flower (Ctrl+C para salir):"
	$(COMPOSE) logs -f flower

# ========== MONITOREO ==========

status:
	@echo "📊 Estado de servicios:"
	$(COMPOSE) ps
	@echo ""
	@echo "🔗 URLs:"
	@echo "   Airflow: http://localhost:8080"
	@echo "   Flower: http://localhost:5555"
	@echo "   PostgreSQL: localhost:5432"
	@echo "   Redis: localhost:6379"

flower:
	@echo "🌸 Abriendo Flower en navegador..."
	@command -v open >/dev/null 2>&1 && open http://localhost:5555 || \
		command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:5555 || \
		command -v start >/dev/null 2>&1 && start http://localhost:5555 || \
		echo "Abre manualmente: http://localhost:5555"

airflow:
	@echo "🌐 Abriendo Airflow UI en navegador..."
	@command -v open >/dev/null 2>&1 && open http://localhost:8080 || \
		command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:8080 || \
		command -v start >/dev/null 2>&1 && start http://localhost:8080 || \
		echo "Abre manualmente: http://localhost:8080"

# ========== BASE DE DATOS ==========

db-shell:
	@echo "🗄️ Conectando a PostgreSQL..."
	$(COMPOSE) exec postgres psql -U airflow -d airflow

redis-cli:
	@echo "🔴 Conectando a Redis..."
	$(COMPOSE) exec redis redis-cli

# ========== LIMPIAR ==========

clean:
	@echo "⚠️  ¡Este comando borrará TODO!"
	@read -p "¿Estás seguro? (s/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Ss]$$ ]]; then \
		echo "🗑️ Borrando datos..."; \
		$(COMPOSE) down -v; \
		echo "✅ Limpiado. Ejecuta 'make up' para reiniciar."; \
	else \
		echo "Cancelado."; \
	fi

# ========== ATAJOS ==========

test-simple:
	@echo "🧪 Test simple: verificar que DAG se carga"
	$(COMPOSE) exec -T airflow-scheduler airflow dags test mlops_pipeline_basic

info:
	@echo "ℹ️ Información del stack:"
	@echo "Executor: CeleryExecutor"
	@echo "Workers: 3 (data, ml, ml-gpu)"
	@echo "Broker: Redis"
	@echo "DB: PostgreSQL"
	@echo "Monitoreo: Flower"
	@echo ""
	@echo "📚 Documentación:"
	@echo "   - MLOPS_README.md"
	@echo "   - MLOPS_VISUAL_GUIDE.md"
	@echo "   - MLOPS_EXAMPLES.py"
	@echo "   - COMMANDS_CHEATSHEET.sh"
	@echo "   - RESUMEN_EJECUTIVO.md"

# Default target
.DEFAULT_GOAL := help
