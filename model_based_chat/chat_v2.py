import cmd
import logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)))))

from user import User
from model_based_chat.cosine_similarity_model import ExtractKeywords
import time
import re

from LaptopRecommender import LaptopRecommender


class chat_v2:
    
    def __init__(self):
        self.prompts = {}
        self.user = User()
        self.extractor = ExtractKeywords()
        self.name = "Laptop Nerd"

    def set_prompts(self):
        """
        The prompts range from generic in the beginning to specific by the end
        """
        self.prompts["greeting"] = f"It is lovely to meet you {self.user.username}. Thanks for choosing me as your laptop guide for the day. Please enter what kind of laptop you're looking for and I will try my best to recommend you a laptop that suits your needs. A few instructions before you begin:\n - Press: Ctrl/Cmd + C if you would like to end the chat at any time.\n - I'm a new chatbot so I'm learning. Please be patient as I ask questions. I will start with generic questions and get more specific"

        self.prompts["start_chat"] = f"{self.user.username}, so tell me what kind of laptop are you looking for. You can describe anything like its brand, your budget, how fast you want it to be, or anything else on your mind."

        self.prompts["goodbye"] = f"I hope you found my recommendations helpful {self.user.username}! Please start the chat again if you want to explore your other laptop preferences."

        self.prompts["filler"] = ["Ohh okay I see. Tell me more about the speed you'd like.", "That's a good choice! Can you tell me more if you prefer a large screen or not?", "Aha! Okay let me think that through"]

        self.prompts["ram_memory"] = ["Hmm so would you like a laptop that's fast or would you rather prefer something slightly slower but cheaper?", "Is 8GB of RAM good or are you thinking of something higher?"]

        self.prompts["display_size"] = "How big do you want your screen size to be? Like 15 inch, 14 inch etc.?"

        self.prompts["processor_tier"] = ["What kind of processor are you looking for? AMD, Ryzen, intel?", "Any specific processor you're interested in? Some examples are intel core, Ryzen, AMD, etc."]

        self.prompts["brand"] = ["Do you have any specific brand in mind?", "Can you tell me more about the brands you like? For example, Apple, Dell, etc"]

        self.prompts["budget"] = ["Any budget in mind?", "Please provide me with a specific dollar value for your budget"]

        self.prompts["high_confidence"] = "You'd love this recommendation: "

        self.prompts["medium_confidence"] = "I think you'll like this laptop: "

        self.prompts["low_confidence"] = "What about this laptop? It doesn't match all your preferences but you could check it out: "

    def get_prompt(self, keyword):
        return self.prompts[keyword]
    
    def remove_prompt(self, keyword):
        """
        We progressively remove prompts so that we can make the questions more specific. This gives
        a sense of conversationality because we aren't necessarily reusing the same prompt.
        """
        prompt_list = self.get_prompt(keyword)
        if len(prompt_list) > 1:
            self.prompts[keyword] = prompt_list[1::]
    
    def update_user_preferences(self, result):
        """
        Parsing through the entities returned by the model and setting user preferences
        """
        for item in result:
            keyword, category, score = item[0], item[1], item[2]
            
            if self.user.get_entities(category) == "":
            ## if the category is already set then we don't change it. NOTE: If we find a better category later on, then we change it that way
                self.user.set_entities(category, keyword)

    def refine_results(self, user_response, predicted_slots, current_slot):
        """
        Heuristics used to detect user preferences if the model detected the category as 'unknown'
        """
        ## refining for budget if no entities have matched yet
        for item in predicted_slots:
            keyword, category, score = item[0], item[1], item[2]

            if category == 'unknown' and str.isnumeric(keyword) and int(keyword) > 100:
                self.user.set_entities("budget", int(keyword))
            
            ## regex for extracting ram memory info
            elif category == "unknown" and "gb" in keyword.lower():
                extract_number = re.search(r'\d+', keyword)
                number = extract_number.group()
                self.user.set_entities("ram_memory", int(number))
                
            elif category == "unknown" and current_slot == "display_size":
                self.user.set_entities("display_size", float(keyword)) 

            elif current_slot == "ram_memory" and "fast" in user_response:
                self.user.set_entities("ram_memory", "16")

            elif category == "brand" and "macbook" in user_response:
                self.user.set_entities("brand", "Apple")

    def detect_slots(self, user_response, current_slot):
        """
        Main method for parsing user response, classifying it, and returning the result 
        """

        result = self.extractor.classify_tokens(user_response)
        self.update_user_preferences(result)
        self.refine_results(user_response, result, current_slot)
        
        print(f" Backend --- updated user preferences are: {self.user.return_all_entities()}")

    def get_empty_slots(self):
        """
        Returns a list of currently empty slots after each turn
        """
        result = []
        slots = ['brand', 'budget', 'processor_tier', 'display_size', 'ram_memory']

        for slot in slots:
            if self.user.get_entities(slot) == '':
                result.append(slot)

        return result
    
    def processor_preference(self, input):
        """
        Uses binary classification by implementing sentence-level transformer
        to detect whether the user's response said yes or no
        """
        result = self.extractor.get_yes_no_label(input)

        if result == "no":
            self.user.set_entities('processor_tier', " ")

        return result
    
    def set_default_processor(self):
        """
        If a user doesn't care about processor types then we choose the 
        most frequently occuring processors within our dataset as a default
        """
        preferred_brand = self.user.get_entities("brand")
        
        if preferred_brand == r"[A|a]pple":
            self.user.set_entities("processor_tier", "m1")
        else:
            self.user.set_entities("processor_tier", "core i5") # most frequent one in our dataset
    
    def clarify_intent(self, input):
        """
        Clarifies whether this feature is important to the user or not. If it isn't then
        the algorithm choose a default value for that feature in set_default_value
        """
        return self.extractor.get_yes_no_label(input)
    
    def generate_recommendation(self):
        """
        This method calls the LaptopRecommender algorithm to get the predicted top 5 laptops
        """
        recommender = LaptopRecommender(self.user)
        result = recommender.recommend(self.user.return_all_entities())
        return result

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
        self.bot_says("Hello! Welcome to the laptop reccommender chatbot! To start, please enter your username below.")
        username = input("Username: ")
        self.chatbot.user.set_username(username)

        # set the prompts after i got the username
        self.chatbot.set_prompts()

        # communicating the rules
        greeting = self.chatbot.get_prompt("greeting")
        self.bot_says(greeting)
        input("Press any key to continue.")
        self.prompt_user()


    def prompt_user(self):
        # beginning the chat
        start_prompt = self.chatbot.get_prompt("start_chat")
        self.bot_says(start_prompt)

        user_response = input(f"{self.chatbot.user.username}> ")

        self.chatbot.detect_slots(user_response, "")

        previous_slot = ""

        while True: 
            ## get the first empty slot
            if len(self.chatbot.get_empty_slots()) != 0:
                current_slot = self.chatbot.get_empty_slots()[0]
                #print(f"current slot is {current_slot}")

                if current_slot == "processor_tier":
                    self.bot_says("Do you have a preference for processor type or nope? Processors are components that determine the performance and capabilities of the laptop.")
                    user_input = input(f"{self.chatbot.user.username}> ")
                    sentiment_response = self.chatbot.processor_preference(user_input)
                    if sentiment_response == 'no':
                        self.chatbot.set_default_processor()
                        continue

                questions = self.chatbot.get_prompt(current_slot)

                if isinstance(questions, list):
                    questions = questions[0]

                self.bot_says(questions)

                ## call the detect slot
                user_response = input(f"{self.chatbot.user.username}> ")

                self.chatbot.detect_slots(user_response, current_slot)

                if current_slot in self.chatbot.get_empty_slots():
                    self.chatbot.remove_prompt(current_slot)
            else:
                result = self.chatbot.generate_recommendation()
                self.show_recommendations(result)
                break
    
    def show_recommendations(self, result):
        def confident_speech(result):
            score = result['similarity_score']

            if score > 0.8:
                self.bot_says(self.chatbot.get_prompt("high_confidence")+ result['Model'])
            elif score > 0.5 and score < 0.8: 
                self.bot_says(self.chatbot.get_prompt("medium_confidence")+ result['Model'])
            elif score < 0.5:
                self.bot_says(self.chatbot.get_prompt("low_confidence")+ result['Model'])

        length_of_result = len(result) 

        self.bot_says(f"I found {length_of_result} laptops that suit your preferences. Would you like to see the first one?")
        input(f"Please press enter to view them")

        for i in range(0, length_of_result):
            confident_speech(result.iloc[i])
            print(f"\nIts features are:\n -RAM Memory: {result.iloc[i]['ram_memory']}\n -Price {result.iloc[i]['price (usd)']}\n -Processor: {result.iloc[i]['processor_tier']}\n -Display Size: {result.iloc[i]['display_size']}")
            print(f"My confidence for this recommendation is {result.iloc[i]['similarity_score']}")
            input(f"Please press enter to see the next one")
        
        self.bot_says(self.chatbot.get_prompt("goodbye"))
        return

    def bot_says(self, response):
        print(self.bot_prompt + response)
    
    def set_asking_question(self):
        self.asking_question = not (self.asking_question)
    

my_chatbot = InteractionLoop()
my_chatbot.cmdloop()