# Asistente de Investigación (n8n + Postgres)

Este proyecto implementa en n8n el flujo solicitado: búsqueda de artículos académicos recientes, evaluación y resumen con IA (DeepSeek vía OpenRouter), registro de resultados en Postgres, y notificaciones opcionales.

## Variables de entorno
Copia `.env` y ajusta según tu entorno:

```
DATABASE_CLIENT=postgres
DATABASE_HOST=propfirm-vps.elitecode.lat
DATABASE_PORT=5424
DATABASE_NAME=db
DATABASE_USERNAME=postgre
DATABASE_PASSWORD=postgre
DATABASE_SSL=false

FLOW_DEFAULT_FROM_DAYS=30
SEMANTIC_SCHOLAR_API_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=deepseek-chat
SLACK_WEBHOOK_URL=
```

## Base de datos
Ejecuta `db.sql` en tu Postgres de destino para crear la tabla `articles`.

## Importar el workflow en n8n
1. Abre n8n y ve a *Workflows* → *Import*.
2. Importa `n8n-workflows/research_assistant.json`.
3. Crea las credenciales **Postgres-ENV** (Settings → Credentials → Postgres) con:
   - Host: `${DATABASE_HOST}`
   - Port: `${DATABASE_PORT}`
   - Database: `${DATABASE_NAME}`
   - User: `${DATABASE_USERNAME}`
   - Password: `${DATABASE_PASSWORD}`
   - SSL: `${DATABASE_SSL}` (si tu instancia requiere SSL, ajústalo a `true`).
   Nota: Los valores se toman del entorno del servidor donde corre n8n.
4. (Opcional) Si vas a usar Semantic Scholar, añade `SEMANTIC_SCHOLAR_API_KEY` en tu entorno.
5. Añade `OPENROUTER_API_KEY` en tu entorno de n8n para la evaluación IA (modelo `OPENROUTER_MODEL`, por defecto `deepseek-chat`).
6. (Opcional) Configura `SLACK_WEBHOOK_URL` para recibir alertas cuando el artículo sea relevante y contenga términos clave.

## Uso
- Endpoint de disparo (Webhook): `POST /research/search`
- Cuerpo esperado (JSON):
  ```json
  {
    "topic": "tema a buscar",
    "from": "YYYY-MM-DD",
    "to": "YYYY-MM-DD"
  }
  ```
- Si no especificas fechas, usa el último mes (`FLOW_DEFAULT_FROM_DAYS`).
- El flujo consultará Crossref y Semantic Scholar, normalizará los resultados, evaluará cada artículo con IA y guardará en `articles`.

## Notas
- Si `SEMANTIC_SCHOLAR_API_KEY` no está definido, ese nodo puede fallar; el flujo seguirá con Crossref.
- La credencial Postgres no puede importarse en JSON por seguridad; debes crearla en n8n y nombrarla **Postgres-ENV** para que el workflow la reconozca.
- Para otros destinos (Zotero/Mendeley, correo), puedes añadir nodos y usar variables de entorno correspondientes.
- El documento original pedía consumo por Streamlit y Supabase; aquí se adapta a **Postgres** y **toda la funcionalidad en n8n**, conforme a tu indicación.