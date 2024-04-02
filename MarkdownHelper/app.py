import gradio as gr
import json

max_textboxes = 10
visible_textboxes = 0
textboxes_info = []

def get_text(content:str,
             add_title:bool, 
             title_content:str, 
             add_prefix:bool, 
             prefix_mode:str, 
             prefix_content:str, 
             add_suffix:bool, 
             suffix_mode:str, 
             suffix_content:str, 
             drop_blank_row:bool,
             is_optional:bool):
    if is_optional:
        if content == '':
            return ''
    if drop_blank_row:
        rows = [i for i in content.split('\n') if len(i) > 0]
    else:
        rows = content.split('\n')
    if add_prefix:
        if prefix_mode == "each row":
            rows = [prefix_content + i for i in rows]
        elif prefix_mode =="once only":
            rows[0] = prefix_content + rows[0]
        elif prefix_mode =="one line":
            rows[0] = prefix_content + "\n" + rows[0]
    if add_suffix:
        if suffix_mode == "each row":
            rows = [i + suffix_content for i in rows]
        elif suffix_mode =="once only":
            rows[-1] = rows[-1] + suffix_content
        elif suffix_mode =="one line":
            rows[-1] = rows[-1] + "\n" + suffix_content
    if add_title:
        rows = [str(title_content)] + rows
    return "\n".join(rows)

def add_component(*configs): 
    global visible_textboxes, textboxes_info
    textboxes_info.append(configs)
    visible_textboxes += 1
    return get_boxes()

def pop_component():
    global visible_textboxes, textboxes_info
    visible_textboxes -= 1
    textboxes_info.pop()
    return get_boxes()

def get_boxes():
    boxes = []
    for i in range(visible_textboxes):
        boxes.append(gr.Textbox(label=textboxes_info[i][0], placeholder=textboxes_info[i][1], lines=2, visible=True))
    for i in range(max_textboxes - visible_textboxes):
        boxes.append(gr.Textbox(visible=False))
    return boxes

def show_all_contents(*text):
    texts = [get_text(text[i], *textboxes_info[i][2:]) for i in range(visible_textboxes)]
    return "\n".join([i for i in texts if len(i) > 0])

def title_relate_visible(flag:bool):
    if flag:
        title_mode = gr.Dropdown(["Title"], visible=False)
        title_content = gr.Textbox(label="Title", placeholder="Title Content", visible=True)    
    else:
        title_mode = gr.Dropdown(["Title"], visible=False)
        title_content = gr.Textbox(label="Title", placeholder="Title Content", visible=False)
    return [title_mode, title_content]

def prefix_relate_visible(flag:bool):
    if flag:
        prefix_mode = gr.Dropdown(["each row", "once only", "one line"], label="Mode", visible=True)
        prefix_content = gr.Textbox(label="Prefix", placeholder="Prefix Content", visible=True)   
    else:
        prefix_mode = gr.Dropdown(["each row", "once only", "one line"], label="Mode", visible=False)
        prefix_content = gr.Textbox(label="Prefix", placeholder="Prefix Content", visible=False)   
    return [prefix_mode, prefix_content]

def suffix_relate_visible(flag:bool):
    if flag:
        suffix_mode = gr.Dropdown(["each row", "once only", "one line"], label="Mode", visible=True)
        suffix_content = gr.Textbox(label="Suffix", placeholder="Suffix Content", visible=True) 
    else:
        suffix_mode = gr.Dropdown(["each row", "once only", "one line"], label="Mode", visible=False)
        suffix_content = gr.Textbox(label="Suffix", placeholder="Suffix Content", visible=False)
    return [suffix_mode, suffix_content] 

def load_configs(path:str):
    global textboxes_info, visible_textboxes
    with open(path, 'r') as f:
        configs = json.load(f)
    textboxes_info = configs['textboxes_info']
    visible_textboxes = configs['visible_textboxes']
    return get_boxes()

def save_configs():
    global textboxes_info, visible_textboxes
    configs = {'textboxes_info':textboxes_info,
               'visible_textboxes':visible_textboxes}
    with open('configs.json', 'w') as f:
        json.dump(configs, f)


if __name__ == '__main__':
    
    app = gr.Blocks()
    with app:
        gr.Markdown('''# Text Formator
                Developed by Carzit @ Myrian Framework''')
        
        textboxes = []
        for i in range(max_textboxes):
            textbox = gr.Textbox(visible=False)
            textboxes.append(textbox)

        with gr.Accordion(label="Add New Textbox", open=False):
            with gr.Row():
                label_name = gr.Textbox(label="Label")
                place_holder = gr.Textbox(label="Placeholder")

            with gr.Row():
                add_title = gr.Checkbox(label="Add Title")
                title_mode = gr.Dropdown(["Title"], visible=False)
                title_content = gr.Textbox(label="Title", placeholder="Title Content", visible=False)

            with gr.Row():
                add_prefix = gr.Checkbox(label="Add Prefix")
                prefix_mode = gr.Dropdown(["each row", "once only", "one line"], value="each row", label="Mode", visible=False)
                prefix_content = gr.Textbox(label="Prefix", placeholder="Prefix Content", visible=False)

            with gr.Row():
                add_suffix = gr.Checkbox(label="Add Suffix")
                suffix_mode = gr.Dropdown(["each row", "once only", "one line"], value="each row", label="Mode", visible=False)
                suffix_content = gr.Textbox(label="Suffix", placeholder="Suffix Content", visible=False)

            with gr.Row():
                drop_blank_row = gr.Checkbox(label="Drop Blank Row")
                is_optional = gr.Checkbox(label="Optional")

            configs = [label_name, place_holder, add_title, title_content, add_prefix, prefix_mode, prefix_content, add_suffix, suffix_mode, suffix_content, drop_blank_row, is_optional]
            title_relate = [title_mode, title_content]
            prefix_relate = [prefix_mode, prefix_content]
            suffix_relate = [suffix_mode, suffix_content]

            with gr.Row():
                add_btn = gr.Button("Add")
                del_btn = gr.Button("Delete")
                reset_btn = gr.ClearButton(configs, value="Reset")

            with gr.Accordion(label="Load and Save Configs",open=False):
                load_explorer = gr.File(file_types=['.json'])
                save_btn = gr.Button("Save Configs")

        with gr.Row():
            generate_btn = gr.Button(value='Generate', variant="primary")
            clear_btn = gr.ClearButton(textboxes, variant="primary") 

        with gr.Row():
            result = gr.Textbox(label='Result', show_copy_button=True)

        add_title.change(fn=title_relate_visible, inputs=[add_title], outputs=title_relate)
        add_prefix.change(fn=prefix_relate_visible, inputs=[add_prefix], outputs=prefix_relate)
        add_suffix.change(fn=suffix_relate_visible, inputs=[add_suffix], outputs=suffix_relate)

        add_btn.click(fn=add_component, inputs=configs, outputs=textboxes)
        del_btn.click(fn=pop_component, inputs=[], outputs=textboxes)

        load_explorer.change(load_configs, inputs=[load_explorer], outputs=textboxes)
        save_btn.click(save_configs, inputs=[], outputs=[])

        generate_btn.click(fn=show_all_contents, inputs=textboxes, outputs=[result])
        clear_btn.add(result)


    app.launch(inbrowser=True, share=False)