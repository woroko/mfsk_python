import kivy

from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window

Window.size = (400,130)

from kivy.uix.boxlayout import BoxLayout

class LblTxt(BoxLayout):
    from kivy.properties import ObjectProperty
    theTxt = ObjectProperty(None)

class MyApp(App):

    def build(self):
        self.root = Builder.load_file('guitest.kv')
        return self.root

if __name__ == '__main__':
    MyApp().run()
