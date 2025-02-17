
import socket
import re
import csv
from datetime import datetime
from threading import Thread, Lock
import time
import os
import git

# Configuration
HOST = "irc.chat.twitch.tv"
PORT = 6667
CHANNELS = ["#ninja", "#auronplay", "#rubius", "#thegrefg", "#tfue", "#shroud", "#pokimane", "#sodapoppin", "#riotgames", "#myth", "#sypherpk", "#nickmercs", "#summit1g", "#amouranth", "#esl_csgo", "#fortnite", "#loltyler1", "#bugha", "#montanablack88", "#dakotaz", "#drlupo", "#nickeh30", "#rocketleague", "#gotaga", "#gaules", "#faker", "#castro_1021", "#asmongold", "#fernanfloo", "#trymacs", "#syndicate", "#lirik", "#lolitofdez", "#loserfruit", "#wtcn", "#nightblue3", "#anomaly", "#imaqtpie", "#easportsfifa", "#cellbit", "#kendinemuzisyen", "#lilypichu", "#markiplier", "#rainbow6", "#yoda", "#twitch", "#bratishkinoff", "#solaryfortnite", "#unlostv", "#captainsparklez", "#bobross", "#boxbox", "#cdnthe3rd", "#gosu", "#cizzorz", "#yassuo", "#gamesdonequick", "#nadeshot", "#warframe", "#overwatchleague", "#jukes", "#doublelift", "#izakooo", "#rewinside", "#jahrein", "#joshog", "#forsen", "#mithrain", "#greekgodx", "#eleaguetv", "#scarra", "#rakin", "#sovietwomble", "#trick2g", "#gronkh", "#nl_kripp", "#alinity", "#kingrichard", "#cohhcarnage", "#dyrus", "#goldglove", "#ungespielt", "#voyboy", "#pashabiceps", "#skipnho", "#swiftor", "#callofduty", "#stpeach", "#starladder5", "#a_seagull", "#kittyplays", "#nick28t", "#grimmmz", "#sivhd", "#kinggothalion", "#yogscast", "#amaz", "#xqc", "#ludwig", "#illojuan"]  # Add your channels
ANON_NICK = "justinfan12345"
LOG_FILE = "chat_logs.csv"
UPLOAD_INTERVAL = 5  # 5 minutes (GitHub Actions max: 60m)
MAX_RUN_TIME = 30  # 50 minutes (leaving 10m for cleanup)

# ANSI escape code removal
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
csv_lock = Lock()

class ChatLogger:
    def __init__(self):
        self.sock = socket.socket()
        self.repo = git.Repo(os.getcwd())
        self.last_upload = time.time()
        self.start_time = time.time()
        self.running = True

        # Initialize CSV and Git
        self._init_csv()
        self._git_setup()

    def _git_setup(self):
        if not os.path.exists(LOG_FILE):
            open(LOG_FILE, 'w').close()
        self.repo.git.add(LOG_FILE)

    def _init_csv(self):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Channel", "User", "Message"])

    def connect(self):
        try:
            self.sock.connect((HOST, PORT))
            self.sock.send(f"NICK {ANON_NICK}\r\n".encode('utf-8'))
            self.sock.send(f"JOIN {','.join(CHANNELS)}\r\n".encode('utf-8'))
            print(f"Connected to {len(CHANNELS)} channels")
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.running = False

    def parse_message(self, raw):
        pattern = re.compile(r"^:(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #(\w+) :(.*)")
        return pattern.match(raw).groups() if pattern.match(raw) else (None, None, None)

    def log_message(self, channel, user, message):
        clean_msg = ansi_escape.sub('', message).strip()
        timestamp = datetime.now().isoformat()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {channel} | {user}: {clean_msg}")
        with csv_lock:
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([timestamp, channel, user, clean_msg])

    def upload_to_git(self):
        try:
            self.repo.git.add(LOG_FILE)
            if self.repo.index.diff("HEAD"):
                self.repo.git.commit(
                    "-m", 
                    f"Auto-update chat logs {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                origin = self.repo.remote(name='origin')
                origin.push()
                print(f"Uploaded logs at {time.time() - self.start_time:.1f}s")
        except Exception as e:
            print(f"Git error: {str(e)}")

    def auto_uploader(self):
        while self.running:
            if time.time() - self.last_upload > UPLOAD_INTERVAL:
                self.upload_to_git()
                self.last_upload = time.time()
            time.sleep(10)

    def listen(self):
        buffer = ""
        upload_thread = Thread(target=self.auto_uploader)
        upload_thread.start()

        try:
            while self.running and (time.time() - self.start_time < MAX_RUN_TIME):
                buffer += self.sock.recv(4096).decode('utf-8', errors='ignore')
                while '\r\n' in buffer:
                    line, buffer = buffer.split('\r\n', 1)
                    
                    if line.startswith('PING'):
                        self.sock.send("PONG :tmi.twitch.tv\r\n".encode('utf-8'))
                        continue
                        
                    user, channel, message = self.parse_message(line)
                    if user and channel and message:
                        self.log_message(channel, user, message)
                        
        finally:
            self.running = False
            self.upload_to_git()  # Final upload
            self.sock.close()
            upload_thread.join()
            print("Session ended")

if __name__ == "__main__":
    logger = ChatLogger()
    logger.connect()
    logger.listen()
