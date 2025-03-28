apt install xvfb -y -qq
python -m venv venv
source venv/bin/activate


pip install -r requirements.txt
pip install .
git clone https://github.com/tinker495/humenv.git
pip install humenv/.