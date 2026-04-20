import serial
import struct
import csv
from datetime import datetime

PACKET_SIZE = 149
HEADER = bytes([0xAA, 0xBB, 0xCC, 0xDD])


def compute_checksum(data):
    cs = 0
    for b in data:
        cs ^= b
    return cs


ser = serial.Serial('COM3', 115200, timeout=2)
ser.reset_input_buffer()

filename = f"imu_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)

    header_row = ['packet_id']
    for i in range(1, 6):
        header_row.append(f'timestamp_s{i}')
    for i in range(1, 6):
        header_row += [f'gx_s{i}', f'gy_s{i}', f'gz_s{i}']
    for i in range(1, 6):
        header_row += [f'ax_s{i}', f'ay_s{i}', f'az_s{i}']
    writer.writerow(header_row)

    print(f"Salvare in {filename}")
    print("Apasa Ctrl+C pentru oprire")

    packets = 0
    checksum_errors = 0
    buffer = bytearray()

    try:
        while True:
            new_data = ser.read(ser.in_waiting or 1)
            if new_data:
                buffer.extend(new_data)

            while len(buffer) >= PACKET_SIZE:
                idx = buffer.find(HEADER)
                if idx == -1:
                    print(f"Header negasit, buffer size: {len(buffer)}")
                    buffer = buffer[-3:]
                    break
                if idx > 0:
                    print(f"Aruncat {idx} bytes inainte de header")
                    buffer = buffer[idx:]
                    continue
                if len(buffer) < PACKET_SIZE:
                    break

                packet = buffer[:PACKET_SIZE]
                buffer = buffer[PACKET_SIZE:]

                print(f"Pachet complet ({len(packet)} bytes):")
                print(' '.join(f'{b:02X}' for b in packet))

                cs_calc = compute_checksum(packet[:-1])
                cs_recv = packet[-1]
                print(f"Pachet gasit | cs_calc: {cs_calc}, cs_recv: {cs_recv}, match: {cs_calc == cs_recv}")

                if cs_calc != cs_recv:
                    checksum_errors += 1
                    buffer = packet[1:] + buffer
                    continue

                payload = packet[4:-1]  # 144 bytes

                packet_id_val = struct.unpack('<I', payload[0:4])[0]
                timestamps = list(struct.unpack('<5I', payload[4:24]))

                gx = list(struct.unpack('<5f', payload[24:44]))
                gy = list(struct.unpack('<5f', payload[44:64]))
                gz = list(struct.unpack('<5f', payload[64:84]))
                ax = list(struct.unpack('<5f', payload[84:104]))
                ay = list(struct.unpack('<5f', payload[104:124]))
                az = list(struct.unpack('<5f', payload[124:144]))

                floats = gx + gy + gz + ax + ay + az

                print(f"Packet ID: {packet_id_val} | T1: {timestamps[0]} us | Gx1: {floats[0]:.3f} dps | Az1: {az[0]:.4f} g")

                row = [packet_id_val] + timestamps + floats
                writer.writerow(row)
                f.flush()

                packets += 1
                print(f"Pachet salvat: {packets} | Erori: {checksum_errors}")

    except KeyboardInterrupt:
        print(f"Oprire. Total: {packets} pachete | Erori: {checksum_errors}")
        ser.close()