mkdir -p ~/.streamlit/
echo "[general]"  > ~/.streamlit/credentials.toml
echo "email = \"gucho35.enp9@gmail.com\""  >> ~/.streamlit/credentials.toml
echo "[theme]
base='dark'
primaryColor='purple'"
echo "[server]"  > ~/.streamlit/config.toml 
echo "headless = true"  >> ~/.streamlit/config.toml
echo "port = $PORT"  >> ~/.streamlit/config.toml
echo "enableCORS = true"  >> ~/.streamlit/config.toml