import base64
import streamlit as st
from pathlib import Path
from st_clickable_images import clickable_images


class UIComponents:
    @staticmethod
    def get_image_as_base64(file_path):
        with Path(file_path).open('rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    @staticmethod
    def create_clickable_images(image_path, title):
        base64_image = UIComponents.get_image_as_base64(image_path)
        return clickable_images(
            [f"data:image/gif;base64,{base64_image}"],
            titles=[title],
            div_style={"display": "flex", "justify-content": "center", "aligin-items": "center"},
            img_style={'cursor': 'pointer', 'transitioin': 'trainsform .3s', 'width' : '300px'}
        )
    