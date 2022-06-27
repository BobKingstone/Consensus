class EvaluatorEntry():
    def __init__(self, id, address):
        self.__id = id
        self.__address = address


    @property
    def id(self):
        return self.__id


    @property
    def address(self):
        return self.__address