class User:
    """
    This class stores information about user preferences that we obtained through 
    the user's answers to our questions

    --------------------------------------------
    Attributes
    - username = name of the user
    - entities = preferences of the user
    """
    def __init__(self):
        self.username = ""
        self.entities = {"brand": "",
                         "ram_memory": "",
                         "display_size": "",
                         "budget": "",
                         "processor_tier":""}

    def set_entities(self, key, value):
        self.entities[key] = value

    def get_entities(self, key):
        return self.entities.get(key)

    def entities_size(self):
        return len(self.entities)
    
    def get_username(self):
        return self.username
    
    def set_username(self, username):
        self.username = username

    def return_all_entities(self):
        return self.entities