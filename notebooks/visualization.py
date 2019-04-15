import xml.etree.ElementTree as ET

class SvgGoBoard:
    BLACK = 1
    WHITE = -1

    # coord_style: 'A1' or '00'
    def __init__(self, width, height, coord_style='A1', rotate=False):
        self.MARGIN = 40
        self.GRID_SIZE = 25
        self.STONE_SIZE = 11
        self.FONT_SIZE = "15"

        self.width = width
        self.height = height
        self.pixel_width = (self.width-1) * self.GRID_SIZE + self.MARGIN * 2  # with a border
        self.pixel_height = (self.height-1) * self.GRID_SIZE + self.MARGIN * 2
        self.board_color = "wheat"

        self.stones = []
        self.squares = []
        self.coord_style = coord_style
        self.rotate = rotate
        return


    def _repr_svg_(self):
        return self.Draw()


    def AddStone(self, x, y, color):
        if self.rotate:
            self.stones.append((y, x, color))
        else:
            self.stones.append((x, y, color))

        
    def AddSquare(self, x, y, number):
        if self.rotate:
            self.squares.append((y, x, number))
        else:
            self.squares.append((x, y, number))
            

    def DrawText(self, parent, x, y, text):
        el = ET.SubElement(parent, "text", x=str(x), y=str(y))
        el.set("font-size", self.FONT_SIZE)
        el.set("font-weight", "lighter")
        el.text = text


    def DrawStone(self, parent, x, y, stone):
        px = self.MARGIN + x * self.GRID_SIZE
        py = self.MARGIN + y * self.GRID_SIZE
        color = "black" if stone == self.BLACK else "white"
        el = ET.SubElement(parent, "circle", cx=str(px), cy=str(py), r=str(self.STONE_SIZE),
                           style="fill:%s;" % color)


    def DrawSquare(self, parent, x, y, number):
        px = self.MARGIN + x * self.GRID_SIZE
        py = self.MARGIN + y * self.GRID_SIZE
        fill = "yellow" if number >=0 else "pink"
        text = str(number) if number >= 0 else "G"
        el = ET.SubElement(parent, "rect", x=str(px-8), y=str(py-8), width=str(16), height=str(16),
                           fill=fill)
        self.DrawText(parent, px-5, py+5, text)
        

    def DrawStar(self, parent, x, y):
        y = self.height + 1 - y
        px = self.MARGIN + (x-1) * self.GRID_SIZE
        py = self.MARGIN + (y-1) * self.GRID_SIZE
        el = ET.SubElement(parent, "circle", cx=str(px), cy=str(py), r=str(2), style="fill:black;")


    def Draw(self):
        svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", version="1.1",
                         height="%s" % self.pixel_height, width="%s" % self.pixel_width)
        root = ET.SubElement(svg, "g", style="fill-opacity:1.0; stroke:black; stroke-width:1;")
        # Board
        ET.SubElement(root, "rect", x="0", y="0", height=str(self.pixel_height),
                      width=str(self.pixel_width), style="fill:%s;" % self.board_color)
        # Grid: vertical lines
        for k in range(self.width):
            ET.SubElement(root, "line", x1=str(self.MARGIN + k*self.GRID_SIZE), y1=str(self.MARGIN),
                          x2=str(self.MARGIN + k*self.GRID_SIZE),
                          y2=str(self.pixel_height - self.MARGIN))
        # Grid: horizontal lines
        for k in range(self.height):
            ET.SubElement(root, "line",
                          x1=str(self.MARGIN), y1=str(self.MARGIN + k*self.GRID_SIZE),
                          x2=str(self.pixel_width - self.MARGIN),
                          y2=str(self.MARGIN + k*self.GRID_SIZE))

        if self.coord_style == 'A1':
            # Coordinates: A to T
            for k in range(self.width):
                text = chr(65 + (k if k < 8 else (k+1)))
                x = self.MARGIN - 4 + self.GRID_SIZE*k
                self.DrawText(root, x=x, y=18, text=text)
                self.DrawText(root, x=x, y=self.pixel_height - 10, text=text)
            # Coordinates: 1 to 19
            for k in range(self.height):
                text = str(k+1)
                y = self.MARGIN + 6 + self.GRID_SIZE*k
                self.DrawText(root, x=7, y=y, text=text)
                self.DrawText(root, x=self.pixel_width-22, y=y, text=text)
        else:
            # X axis: 0 to 18
            for k in range(self.width):
                text = str(k)
                x = self.MARGIN - 4 + self.GRID_SIZE*k
                self.DrawText(root, x=x, y=18, text=text)
                self.DrawText(root, x=x, y=self.pixel_height - 10, text=text)
            # Y axis: 0 to 18
            for k in range(self.height):
                text = str(k)
                y = self.MARGIN + 6 + self.GRID_SIZE*k
                self.DrawText(root, x=7, y=y, text=text)
                self.DrawText(root, x=self.pixel_width-22, y=y, text=text)

        # Stars
        if self.height == 19 and self.width == 19:
            for x, y in [(4, 4), (10, 4), (15, 4), (4, 10), (4, 15), (10, 4), (10, 15),
                         (15, 4), (15, 15)]:
                self.DrawStar(root, x, y)

        if self.height == self.width and self.width % 2 == 1:
            self.DrawStar(root, (self.width + 1) / 2 , (self.width + 1) / 2)

        # Stones
        for x, y, c in self.stones:
            self.DrawStone(root, x, y, c)
            
        # Annotations
        for x, y, n in self.squares:
            self.DrawSquare(root, x, y, n)
            
        return str(ET.tostring(svg))