import pyautogui as pag
import time as t

t.sleep(2)
print(pag.position())

pag.moveTo(736, 623)
t.sleep(0.2)
pag.mouseDown()
t.sleep(0.2)
pag.moveTo(588, 623)
pag.mouseUp()
t.sleep(0.2)
pag.moveTo(635, 623)
t.sleep(0.2)
pag.mouseDown()
pag.moveTo(362, 555)
pag.mouseUp()

#Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's
# standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make
# a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting,
# remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing
# Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions
# of Lorem Ipsum.