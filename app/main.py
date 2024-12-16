import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from search import search_similarity, process_image_for_encoder_gradio
from utils import str_to_bytes
from io import BytesIO

def add_ranking_number(image, rank):
    """Añade un número de ranking a la imagen"""
    img_with_rank = image.copy()
    draw = ImageDraw.Draw(img_with_rank)
    
    width, height = image.size
    circle_radius = min(width, height) // 15
    circle_position = (circle_radius + 10, circle_radius + 10)
    
    draw.ellipse(
        [(circle_position[0] - circle_radius, circle_position[1] - circle_radius),
         (circle_position[0] + circle_radius, circle_position[1] + circle_radius)],
        fill='white',
        outline='black'
    )
    
    font_size = circle_radius
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = str(rank + 1)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = (
        circle_position[0] - text_width // 2,
        circle_position[1] - text_height // 2
    )
    
    draw.text(text_position, text, fill='black', font=font)
    return img_with_rank

def process_image_result(image_str, rank):
    """Convierte una cadena de imagen en un objeto PIL Image con ranking"""
    try:
        img = Image.open(BytesIO(str_to_bytes(image_str)))
        return add_ranking_number(img, rank)
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None
      
def interface_fn(mode, input_text, input_image, top_k):
    try:
        # Determinar qué input usar basado en el modo
        if mode == "text":
            if not input_text.strip():
                return [], "Por favor, ingresa un texto para buscar."
            input_data = input_text
        else:  # mode == "image"
            if input_image is None:
                return [], "Por favor, sube una imagen para buscar."
            input_data = process_image_for_encoder_gradio(input_image, is_bytes=False)

        # Show the input data
        print(f"Input data: {input_data}")  # Para debugging
        
        # Realizar la búsqueda
        results = search_similarity(input_data, mode, int(top_k))
        
        # Formatear resultados según el modo
        if mode == "text":  # Devuelve imágenes
            processed_images = []
            # Si results es una lista de listas, la aplanamos
            if results and isinstance(results[0], list):
                print("Recibida lista de listas, aplanando...")  # Para debugging
                results = [item for sublist in results for item in sublist]
            
            for idx, img_str in enumerate(results):
                img = process_image_result(img_str, idx)
                if img is not None:
                    processed_images.append(img)
            
            if not processed_images:
                return [], "No se pudieron procesar las imágenes"
            return processed_images, None
            
        else:  # mode == "image" - Devuelve textos
            if isinstance(results, list):
                numbered_texts = [f"{i+1}. {text}" for i, text in enumerate(results)]
                return [], "\n\n".join(numbered_texts)
            else:
                return [], str(results)
            
    except Exception as e:
        print(f"Error en interface_fn: {str(e)}")
        print(f"Tipo de resultados: {type(results)}")  # Para debugging
        return [], f"Error durante la búsqueda: {str(e)}"


def search_text(input_text, top_k):
    try:
        if not input_text.strip():
            return []
        
        # Realizar la búsqueda
        results = search_similarity(input_text, "text", int(top_k))
        
        processed_images = []
        # Si results es una lista de listas, la aplanamos
        if results and isinstance(results[0], list):
            results = [item for sublist in results for item in sublist]
        
        for idx, img_str in enumerate(results):
            img = process_image_result(img_str, idx)
            if img is not None:
                processed_images.append(img)
        
        return processed_images
            
    except Exception as e:
        print(f"Error en search_text: {str(e)}")
        return []

with gr.Blocks() as demo:
    gr.Markdown("# Buscador de Similitud por Texto")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texto de búsqueda",
                placeholder="Ingresa aquí tu texto...",
                lines=3
            )
            
            top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Número de resultados",
                info="¿Cuántos resultados similares quieres ver?"
            )
            
            search_button = gr.Button("Buscar")
        
        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Imágenes similares", 
                columns=3,
                height="auto"
            )
    
    search_button.click(
        fn=search_text,
        inputs=[input_text, top_k],
        outputs=output_gallery
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    demo.launch()