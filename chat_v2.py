import cmd
from user import User
from cosine_similarity_model import ExtractKeywords

class chat_v2:
    
    def __init__(self):
        self.prompts = {}
        self.user = User()

    def set_prompt(self, topic):
        self.prompts["greeting"] = f"It is lovely to meet you {self.user.username}. Thanks for choosing me as your laptop guide for the day. Please enter what kind of laptop you're looking for and I will try my best to recommend you a laptop that suits your needs.
        A few instructions before you begin:
        :quit if you would like to end the chat at any time.\n"

        self.prompts["start_chat"] = f"{self.user.username}, so tell me what kind of laptop are you looking for"

        self.prompts["filler"] = ["Ohh okay I see. Tell me more about the speed you'd like.", "That's a good choice! Can you tell me more if you prefer a large screen or not?", "Aha! Okay let me think that through"]

        self.prompts["ram_memory"] = ["Hmm so would you like a laptop that's fast or would you rather prefer something slightly slower but cheaper?", "Is 8GB of RAM good or are you thinking of something higher?"]

        self.prompts["display_size"] = "How big do you want your screen size to be? Like 15 inch, 14 inch etc.?"

        self.prompts["processor_tier"] = ["What kind of processor are you looking for? AMD, Ryzen, intel?", "Any specific processor you're interested in? Some examples are intel core, Ryzen, AMD, etc."]

        self.prompts["brand"] = "Do you have any specific brand in mind?"

        self.prompts["price"] = "Any budget in mind?"
    
    def get_prompt(self, keyword):
        return self.prompts[keyword]
    
    def detect_intent(self, user_response):
        extractor = ExtractKeywords()
        


class InteractionLoop(cmd.Cmd):
    """
    Loop that handles communication with the user. We use command prompt for now.
    
    NOTE: We learned how to have interactionLoop thanks to Heather's code activity in class.
    """
    prompt = '> '

    def __init__(self):
        super().__init__()
        self.chatbot = chat_v2()
        self.name = self.chatbot.name
        self.bot_prompt = '\001\033[96m\002%s> \001\033[0m\002' % self.name
        self.intro_user()
        self.asking_question = True


    def intro_user(self):
        print(self.bot_says("Hello! Welcome to the laptop reccommender chatbot! To start, please enter your username below."))
        username = input("Username: ")
        self.chatbot.user.set_username(username)

        # set the prompts after i got the username
        self.chatbot.set_prompts()

        # communicating the rules
        greeting = self.chatbot.get_prompt("greeting")
        print(self.bot_says(greeting))
        input("Press any key to continue.")
        self.prompt_user()

    def prompt_user(self):
        start_prompt = self.get_prompt("start_chat")
        self.bot_says(start_prompt)

        user_response = input(f"{self.user.username}>")

        self.detect_intent(user_response)




    def bot_says(self, response):
        print(self.bot_prompt + response)
    
    def set_asking_question(self):
        self.asking_question = not (self.asking_question)
    

my_chatbot = InteractionLoop()
my_chatbot.cmdloop()