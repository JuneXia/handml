class Base(object):
    def __init__(self):
        self.num = 0

    def __call__(self, *args, **kwargs):
        print(args)

class Human(Base):
    def __init__(self):
        super(Human, self).__init__()
        print(self.num)

    def __call__(self, *args, **kwargs):
        print(args)


if __name__ == '__main__1':  # 类的继承实验
    person = Human()
    person()

    print('debug')
