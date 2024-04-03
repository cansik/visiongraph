import logging

from openvino.runtime import Core


def get_inference_engine_device() -> str:
    core = Core()

    device_id = 0
    devices = core.available_devices

    # try to find preferred GPU device
    for i, device in enumerate(devices):
        if device.startswith("GPU"):
            device_id = i
            break

    # list all devices
    logging.info(f"OpenVino Devices")
    for i, device in enumerate(devices):
        default_sign = ""

        if i == device_id:
            default_sign = " (Default)"

        full_device_name = core.get_property(device, "FULL_DEVICE_NAME")
        logging.info(f"{device}{default_sign}: {full_device_name}")

    return devices[device_id]
