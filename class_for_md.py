class raport_md:
    
    f = __file__
        
    def __init__(self, file):
        
        self.f = file;
        
    def __del__(self):
        self.f.close()
    

    def text(self, txt):
        self.f.write(txt + '<br>' + '\n')
        
    def text2(self, txt):
        self.f.write(txt)
    
    def title(self, tit):
        self.text2(f"## {tit}\n")
        
    def subtitle(self, subtit):
        self.text2(f"**{subtit}**<br>\n") 
        
    def header(self, head):
        self.text2(f"### {head}\n") 
    
    def new_paragraph(self, tit, txt):
        self.header(tit)
        self.text2(f"{txt}<br>\n")
        
    def ln(self, num):
        for i in range(num):
            self.text('<br>' + '\n')
            
    def nl(self):
        self.text2('\n')
            
    def table(self, names, results):
        self.text2('| Lp. |')
        
        for name in names:
            self.text2(name + ' |')
        self.nl()
        
        self.text2('|:-----------:')
        for i in range(len(names)-1):
            self.text2('|:-----------:')
        
        self.text2('|:-----------:|' + '\n')
        
        for i, row in enumerate(results):
            
            self.text2('| ' + str(i+1) + ' |')
            for res in row:
                res = str(res)
                res = self.check_specials(res)
                self.text2(res + ' |')
                
            self.nl()
        self.nl()
        
    def plot_im(self, im_name, extension):
        self.text2('![plot](' + im_name + extension + ')' + '\n')

    def check_specials(self, string):
        specials = ("!", "#", "*", "-", "_", "+", "'", ".", "(", ")", "[", "]", "{", "}")
        for i, sub in enumerate(specials):
            if sub in string:
                string = string.replace(sub, "\\" + sub)
        
        return string

        
            