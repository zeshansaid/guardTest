INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
        "description": "User's text prompt"
    },
    "image_url": {
        "type": str,
        "required": False,
        "description": "Optional URL to an image for visual question answering"
    }
}
