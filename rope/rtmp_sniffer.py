
import pyshark
import threading
import json
import os
import sys
import time
from rope.language_manager import load_language_dict
Dicts = load_language_dict()
DEFAULT_DATA = Dicts.DEFAULT_DATA
def get_base_dir():
    """获取程序运行时的根目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(get_base_dir(), 'config', 'config.json')
try:
    with open(config_path, 'r', encoding='utf-8') as file:
        json_data = file.read()
        data = json.loads(json_data)
except FileNotFoundError:
    print(f"{DEFAULT_DATA['Error']}: {DEFAULT_DATA['ConfigFileNotFound']} {config_path}")
    data = {'tshark': {'display_filter': '', 'interface': ''}}
except json.JSONDecodeError:
    print(f"{DEFAULT_DATA['Error']}: {DEFAULT_DATA['ConfigFileFormatError']} {config_path}")
    data = {'tshark': {'display_filter': '', 'interface': ''}}
display_filter = data.get('tshark', {}).get('display_filter', '')
interface = data.get('tshark', {}).get('interface', '')
try:
    tshark_path = os.path.join(os.environ['WIRESHARK_PATH'], 'tshark.exe')
except KeyError:
    tshark_path = ''
class RTMPInfoFound(Exception):
    pass
def get_rtmp_info():
    server = None
    code = None
    event = threading.Event()
    packet_count = 0
    capture = None
    def packet_callback(packet):
        nonlocal packet_count
        nonlocal code
        nonlocal server
        try:
            packet_count += 1
            if packet_count % 100 == 0:
                print(f"{DEFAULT_DATA['Processed']} {packet_count} {DEFAULT_DATA['Packets']}...")
            packet_str = str(packet)
            if not server:
                server = filter_strings(packet_str, 'rtmp://')
            if not code:
                code = filter_strings(packet_str, 'stream-')
            if server and code:
                    print(f"{DEFAULT_DATA['Found']} {packet_count} {DEFAULT_DATA['Packets']} {DEFAULT_DATA['RTMPInfo']}")
                    print(f"{DEFAULT_DATA['Server']}: {server}")
                    code = code.replace('\'', '')
                    print(f"{DEFAULT_DATA['StreamKey']}: {code}")
                    event.set()
                    raise RTMPInfoFound()
        except RTMPInfoFound:
            raise
        except Exception as e:
            print(f"{DEFAULT_DATA['Error']}: {str(e)}")
    def filter_strings(input_str, target_str):
        words = input_str.split()
        for word in words:
            if target_str in word:
                return word
        return None
    try:
        try:
            print(DEFAULT_DATA['RTMPSnifferInfoText'])
            capture = pyshark.LiveCapture(interface=interface, display_filter=display_filter, tshark_path=tshark_path)
            capture.apply_on_packets(packet_callback, timeout=60)
        except Exception as e:
            if not server or not code:
                print(f"{DEFAULT_DATA['Error']}: {e}")
    finally:
        if capture:
            capture.close()
    return (server, code)
if __name__ == '__main__':
    print(DEFAULT_DATA['StartRTMPSniffer'])
    print(f"{DEFAULT_DATA['UsedInterface']}: {interface}")
    print(f"{DEFAULT_DATA['UsedDisplayFilter']}: {display_filter}")
    print(f"{DEFAULT_DATA['UsedTsharkPath']}: {tshark_path}")
    start_time = time.time()
    server, code = get_rtmp_info()
    end_time = time.time()
    print(f"{DEFAULT_DATA['TimeConsumed']}: {end_time - start_time:.2f} {DEFAULT_DATA['Seconds']}")
    if server and code:
        print(f"{DEFAULT_DATA['SuccessRTMPInfo']}")
        print(f"{DEFAULT_DATA['Server']}: {server}")
        print(f"{DEFAULT_DATA['StreamKey']}: {code}")
    else:
        if server:
            print(f"{DEFAULT_DATA['OnlyServerInfo']}: {server}")
            print(f"{DEFAULT_DATA['NoStreamKeyInfo']}")
        else:
            if code:
                print(f"{DEFAULT_DATA['OnlyStreamKeyInfo']}: {code}")
                print(f"{DEFAULT_DATA['NoServerInfo']}")
            else:
                print(f"{DEFAULT_DATA['NoRTMPInfo']}")