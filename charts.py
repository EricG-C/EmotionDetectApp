import pandas as pd
import circlify
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import os
import mpld3

def make_bubble_graph(data,detection_of_text):

    sorted_data = {}
    sorted_keys = sorted(data, key=data.get)[-5:]

    for w in sorted_keys:
        sorted_data[w] = float(data[w]) if float(data[w]) > 0 else 0.1

    circles = circlify.circlify(
    sorted_data.values(), 
    show_enclosure=False, 
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )  

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title(f'Top 5 {detection_of_text} distribution')
    ax.set_axis_off()
    ax.set_facecolor("#0d1422")
    labels = sorted_data.keys()
    colors = ['#FF5733', '#C70039', '#900C3F', '#581845', '#2C3E50', '#7D6608', '#7E5109', '#784212', '#512E5F', '#283747', '#154360', '#0E6251', '#145A32', '#1B4F72', '#7E5109', '#784212', '#2E4053', '#2C2C54', '#8E44AD']

    lim = max(
    max(
        abs(circle.x) + circle.r,
        abs(circle.y) + circle.r,
    )
    for circle in circles
    )   

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    for circle, label, color in zip(circles, labels, colors):
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=0.8, linewidth=2, color=color))
        plt.annotate(
            label, 
            (x,y ) ,
            va='center',
            ha='center'
        )
    encoded_img_data = mpld3.fig_to_html(fig)
    return encoded_img_data