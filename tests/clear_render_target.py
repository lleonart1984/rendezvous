from rendering.manager import *
from glm import *
import time

presenter = create_presenter(800, 600, Format.UINT_BGRA_STD, PresenterMode.SDL,
                             usage=ImageUsage.RENDER_TARGET | ImageUsage.TRANSFER_DST, debug=False)

window = presenter.get_window()

last_time = time.perf_counter()
fps = 0
while True:
    fps += 1
    if time.perf_counter() - last_time >= 1:
        last_time = time.perf_counter()
        print("FPS: %s" % fps)
        fps = 0

    event, args = window.poll_events()
    if event == Event.CLOSED:
        break

    with presenter as p:
        with p.get_graphics() as man:
            man.clear_color(p.render_target(), vec4(1,0.4,0.5,1))