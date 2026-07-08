FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY src/street_sign_project/streamlit_app.py src/street_sign_project/streamlit_app.py

ENV PATH="/root/.local/bin:${PATH}"
RUN uv tool install streamlit --with requests

ENV PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_LOCK_API_URL=true

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "streamlit run src/street_sign_project/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT}"]
