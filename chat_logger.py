import socket
import re
import csv
from datetime import datetime
import time
import os

# Configuration
HOST = "irc.chat.twitch.tv"
PORT = 6667
CHANNELS = ["#ninja", "#auronplay", "#rubius", "#thegrefg", "#tfue", "#shroud", "#pokimane", "#sodapoppin", "#riotgames", "#myth", "#sypherpk", "#nickmercs", "#summit1g", "#amouranth", "#esl_csgo", "#fortnite", "#loltyler1", "#bugha", "#montanablack88", "#dakotaz", "#drlupo", "#nickeh30", "#rocketleague", "#gotaga", "#gaules", "#faker", "#castro_1021", "#asmongold", "#fernanfloo", "#trymacs", "#syndicate", "#lirik", "#lolitofdez", "#loserfruit", "#wtcn", "#nightblue3", "#anomaly", "#imaqtpie", "#easportsfifa", "#cellbit", "#kendinemuzisyen", "#lilypichu", "#markiplier", "#rainbow6", "#yoda", "#twitch", "#bratishkinoff", "#solaryfortnite", "#unlostv", "#captainsparklez", "#bobross", "#boxbox", "#cdnthe3rd", "#gosu", "#cizzorz", "#yassuo", "#gamesdonequick", "#nadeshot", "#warframe", "#overwatchleague", "#jukes", "#doublelift", "#izakooo", "#rewinside", "#jahrein", "#joshog", "#forsen", "#mithrain", "#greekgodx", "#eleaguetv", "#scarra", "#rakin", "#sovietwomble", "#trick2g", "#gronkh", "#nl_kripp", "#alinity", "#kingrichard", "#cohhcarnage", "#dyrus", "#goldglove", "#ungespielt", "#voyboy", "#pashabiceps", "#skipnho", "#swiftor", "#callofduty", "#stpeach", "#starladder5", "#a_seagull", "#kittyplays", "#nick28t", "#grimmmz", "#sivhd", "#kinggothalion", "#yogscast", "#amaz", "#xqc", "#ludwig", "#illojuan", "#Caedrel", "#knekro", "#buster", "#s1mple", "#Kamet0", "#otplol_", "#LEC", "TheRealMarzaa", "#MontanaBlack88", "#TUITÊNBÔ", "#eliasn97", "#PGL", "#Gaules", "#sasavot"]  # Add your channels here
ANON_NICK = "justinfan12345"
MAX_RUN_TIME = 2000  # 50 minutes (GitHub Actions limit: 60m)
MESSAGE_DELAY = 0  # Anti-flood delay
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
LOG_FILE_BASE = "chat_logs"
LOG_FILE_EXT = ".csv"

# ANSI escape code removal
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class TwitchChatLogger:
    def __init__(self):
        self.sock = socket.socket()
        self.running = True
        self.start_time = time.time()
        self.file_counter = 1
        self.log_file = self._get_log_file_name()
        self._init_csv()

    def _get_log_file_name(self):
        """Generate log file name with counter."""
        return f"{LOG_FILE_BASE}{self.file_counter}{LOG_FILE_EXT}"

    def _init_csv(self):
        """Initialize CSV only if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Channel", "User", "Message"])

    def _check_file_size(self):
        """Check file size and create new file if needed."""
        if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > MAX_FILE_SIZE:
            self.file_counter += 1
            self.log_file = self._get_log_file_name()
            self._init_csv()
            print(f"Created new log file: {self.log_file}")

    def connect(self):
        """Connect to Twitch IRC anonymously"""
        try:
            self.sock.connect((HOST, PORT))
            self.sock.send(f"NICK {ANON_NICK}\r\n".encode('utf-8'))
            self.sock.send(f"JOIN {','.join(CHANNELS)}\r\n".encode('utf-8'))
            print(f"Connected to {len(CHANNELS)} channels")
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.running = False

    def parse_message(self, raw):
        """Parse IRC message into components"""
        pattern = re.compile(
            r"^:(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #(\w+) :(.*)"
        )
        match = pattern.match(raw)
        return match.groups() if match else (None, None, None)

    def log_message(self, channel, user, message):
        """Append message to CSV with timestamp"""
        try:
            clean_msg = ansi_escape.sub('', message).strip()
            timestamp = datetime.now().isoformat()

            # Print to console for GitHub Actions logging
            print(f"[{timestamp}] {channel} | {user}: {clean_msg}")

            # Append to CSV
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, channel, user, clean_msg])

            # Check and create new file
            self._check_file_size()

        except Exception as e:
            print(f"Logging error: {str(e)}")

    def run(self):
        """Main processing loop"""
        buffer = ""
        try:
            while self.running and (time.time() - self.start_time < MAX_RUN_TIME):
                buffer += self.sock.recv(4096).decode('utf-8', errors='ignore')

                while '\r\n' in buffer:
                    line, buffer = buffer.split('\r\n', 1)

                    # Handle PING/PONG
                    if line.startswith('PING'):
                        self.sock.send("PONG :tmi.twitch.tv\r\n".encode('utf-8'))
                        print("[System] Responded to PING")
                        continue

                    # Parse and log messages
                    user, channel, message = self.parse_message(line)
                    if user and channel and message:
                        self.log_message(channel, user, message)

                    time.sleep(MESSAGE_DELAY)

        except Exception as e:
            print(f"Runtime error: {str(e)}")
        finally:
            self.sock.close()
            print("Connection closed")

if __name__ == "__main__":
    logger = TwitchChatLogger()
    logger.connect()
    logger.run()
    print(f"Completed {MAX_RUN_TIME//60}m session")
