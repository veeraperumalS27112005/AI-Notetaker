import sounddevice as sd
devices = sd.query_devices()
print("=== All Audio Devices ===")
for i, d in enumerate(devices):
    tag = ""
    name_lower = d['name'].lower()
    if d['max_input_channels'] > 0:
        if any(k in name_lower for k in ["cable output", "stereo mix", "loopback"]):
            tag = " <<<< BEST FOR SYSTEM CAPTURE"
        elif d['max_input_channels'] > 0:
            tag = " (input capable)"
    print(f"  [{i}] {d['name']} | in={d['max_input_channels']}  out={d['max_output_channels']}{tag}")

print("\n=== Default devices ===")
print(f"  Default input  : {sd.default.device[0]}")
print(f"  Default output : {sd.default.device[1]}")

# Simulate auto-detection
preferred = ["cable output", "stereo mix", "what u hear", "loopback"]
found = None
for priority_name in preferred:
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if priority_name in name and dev['max_input_channels'] > 0:
            found = (i, dev['name'])
            break
    if found:
        break

if found:
    print(f"\n>>> Auto-detect will use: [{found[0]}] {found[1]}")
else:
    di = sd.default.device[0]
    print(f"\n>>> Auto-detect will fall back to default input: [{di}] {devices[di]['name']}")
