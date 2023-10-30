from nicegui import ui

dark = ui.dark_mode().enable()

with ui.splitter() as splitter:
    with splitter.before:
        ui.code('''
                    (set-logic LIA)

                    (synth-fun f ((x Int) (y Int)) Int)
                    
                    (declare-var x Int)
                    (declare-var y Int)
                    (constraint (= (f x y) (f y x)))
                    (constraint (and (<= x (f x y)) (<= y (f x y))))
                    
                    (check-synth)
                ''').classes('w-full')
    with splitter.after:
        editor = ui.editor(placeholder='Type something here')
        ui.markdown().bind_content_from(editor, 'value',
                                        backward=lambda v: f'HTML code:\n```\n{v}\n```')

ui.run()
