class LoadingBar:
    def __init__(self, upper, size=50, title="", interval=1) -> None:
        self.curr = 0
        self.upper = upper
        self.size = size
        self.title = title
        self.interval = interval
        self.spinner  = '⟿⤳↝⤴⤺⤹⤿↺⥀⟲⭘⭙⟴⇴' 
    def update(self, titleUpdate=None):
        
        if titleUpdate is not None:
            if len(titleUpdate) < len(self.title):
                for _ in range(len(self.title) - len(titleUpdate)):
                    titleUpdate += " "
            self.title = titleUpdate
        self.curr += 1
        if self.curr % self.interval != 0:
            return
        spinIndex = (self.curr // self.interval) % len(self.spinner)
        complete = int((self.curr / self.upper) * self.size)
        s = self.title + "\t═╣"
        for i in range(complete):
            s += "▒"
        for i in range(self.size - complete):
            s += "-"
        s += "╠═ " + str(self.curr) + "/" + str(self.upper) + " - "
        s += "%0.2f" % ((self.curr * 100) / self.upper)
        s += "% "
        s += self.spinner[spinIndex]
        print(s, end="\r")
    
    def adjustUpper(self, diff):
        self.upper += diff

    def complete(self):
        self.curr = self.upper - 1
        self.interval = 1
        self.update()
        print("")
