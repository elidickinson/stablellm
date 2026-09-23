The Dockerfile does `COPY *.py`, so every top-level module ships without a Dockerfile edit. tests/ and pi-extension/ are not copied.
