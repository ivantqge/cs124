# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
import openai
import random

import numpy as np
#import nltk
import re
#from nltk.corpus import stopwords
#nltk.download("punkt")
from porter_stemmer import PorterStemmer

class Translator(BaseModel):
        Translation: str = Field(default="")
            
class SentimentExtractor(BaseModel):
        Anger: bool = Field(default=False)
        Disgust: bool = Field(default=False)
        Fear: bool = Field(default=False)
        Happiness: bool = Field(default=False)
        Sadness: bool = Field(default=False)
        Surprise: bool = Field(default=False)

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'movierobot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.ratings = self.binarize(ratings)
        self.stemmer = PorterStemmer()
        self.negation_words = set([ "not", "no", "never", "none", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "isn't", "aren't", "wasn't", "weren't", "can't", "couldn't", "shouldn't", "won't", "wouldn't", "don't", "doesn't", "haven't", "hasn't", "hadn't", "without", "didn't" ]) 
        # Clause boundaries that reset negation 
        self.clause_boundaries = {',', ';', '.'}
        
        self.user_ratings = np.zeros(9125)
        self.recommendations = []
        self.num_inputted = 0
        self.recommendations_given = 0
        self.recommending = False
        
        self.persona_prompt = """ Take in the given response and make it sound like SpongeBob would say it. Some of these might include 'I'm ready!', 'Order up!', 'Aye-aye, captain!', 'Barnacles!', 'Tartar sauce!', 'Fish paste!', or 'Hoppin' clams!' Remember to keep the main message and most off the words in the input. Return only your converted answer. If there are words in quotations, such as "Hello", remember to keep that part of the input in your answer and do not remove it, including the quotes as well. Don't add anything in between the quotes. Give only two sentences maximum.  
        """


        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = f"Hi! Welcome to the world's first movie recommendation chatbot! My name is {self.name}. How can I help you?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Enjoy your movie! Come back to see me soon. Have a nice day!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = f"""Your name is {self.name}. You are a movie recommender chatbot designed to help users find movies they like and provide information about movies. When a user mentions a movie and their sentiment towards it, acknowledge their sentiment and the movie, then encourage them to discuss another movie, e.g., "Ok, you liked 'The Notebook'! Tell me what you thought of another movie." Always keep the conversation focused on movies. If a user attempts to change the subject, politely steer them back, e.g., "As a moviebot assistant, my job is to help you with only your movie-related needs! Anything film-related that you'd like to discuss?" After the user has discussed five different movies with you, automatically ask if they would like a movie recommendation, e.g., "Ok, now that you've shared your opinion on 5/5 films, would you like a recommendation?" Your primary function is to engage users in conversations about movies, understand their preferences through these discussions, and use this information to make tailored movie recommendations."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        
        if self.recommending:
            if line == "no" or line == "No" or line == "nah" or line == "nope" or line == "Nope" or line == "Nah":
                return "Goodbye! Please enter :quit to leave"
            else:
                self.recommendations_given += 1 
                recommended_movie = self.recommendations[self.recommendations_given]
                response_templates = [
                "I recommend {}. Interested in another suggestion?",
                "{} might be great movie recommendation for you. Would you like to hear about another recommendation?",
                "Given what you told me, I think you would like {}.? Shall I suggest another recommendation?",
                "You might enjoy {}. Want more recommendations?",
                "Let's go with {} for a recommendation this time. Another one?",
                "{} could be a great recommendation. Need another recommendation?",
                ]
                selected_template = random.choice(response_templates)
                formatted_response = selected_template.format('"' + self.titles[recommended_movie][0] + '"')
                self.recommending = True
                if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, formatted_response)
                return formatted_response
                
        preprocessed_input = self.preprocess(line)
        movie_titles = self.extract_titles(preprocessed_input)
        if len(movie_titles) == 1:
            response = self.process_movie_titles(movie_titles, preprocessed_input)
        elif len(movie_titles) > 1:
            if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, "Please tell me about one movie at a time. Go ahead.")
            return "Please tell me about one movie at a time. Go ahead."
        else:
            if self.llm_enabled:
                system_prompt = """If there is no significant emotion in the text (multiple words expressing a certain emotion), output some generic response telling the user that this is not what you are interested in and want to just talk about movies such as 'Hm, that's not really what I want to talk about right now, let's go back to movies' or give a generic response like 'Ok, got it.' Don't give both. If there is significant emotion, output an appropriate response responding to that emotion. For example, if the user says they are angry, output something like 'I am sorry, why are you angry?'. Demonstrate the detection of the expressed emotions in your response. Limit the output to 2 sentences maximum. """
                stop = ["\n"]
                message = preprocessed_input
                response = util.simple_llm_call(system_prompt, message, stop = stop)
                if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, message)
                return response
            resp = ["Can you talk about movies please?", 
                    "Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've                      seen recently?", 
                    "I am sorry, but I didn't catch the name of any movie in your message. Could you mention a film you                              enjoyed?", 
                    "It seems I missed the movie reference there. Could you share details of a recent film you enjoyed?", "I'd                      love to hear about movies. What's your take", 
                    "Let's talk about movies. What are your favorites?", 
                    "I didn't pick up a movie title in your response. Can you recommend a film you like?"]
            response = random.choice(resp)
            if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, response)
            return response
        
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        
        #if self.llm_enabled:
        #    response = "I processed {} in LLM Programming mode!!".format(line)
        #else:
        #    response = "I processed {} in Starter (GUS) mode!!".format(line)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response
    
    def process_movie_titles(self, movie_titles, preprocessed_input):
        movie_index = self.find_movies_by_title(movie_titles[0])
        if len(movie_index) > 1:
            return 'Could you specify which version of the "' + movie_titles[0] + '" you are referring to by adding the year?' 
        elif len(movie_index) == 0:
            resp = [
                    "I've never heard of {}. Can you tell me about a different movie?",
                    "I am sorry, but I am not familiar with {}. Is there a different movie that you like?",
                    "I don't think I know {}. Tell me about a different movie.",
                    "I am sorry, but I do not know much about {}. Can you tell me about another movie?"
                 ]
            formatted_resp = [response.format('"' + movie_titles[0] + '"') for response in resp]
            response = random.choice(formatted_resp)
            if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, response)
            return response
        else:
            sentiment = self.extract_sentiment(preprocessed_input)
            if (sentiment == 0):
                movie_title = '"' + movie_titles[0] + '"'
                resp = [
                "I'm not sure how you feel about {}. Can you clarify?",
                "I'm sorry, I'm not sure if you liked {}. Tell me more about it.",
                "Could you elaborate on your thoughts about {}? I didn't quite catch your opinion.",
                "It's unclear to me whether you enjoyed {}. Could you provide more details?",
                "Your feelings about {} aren't clear to me. Could you elaborate?",
                "Can you explain a bit more about your view on {}? I'm interested in your thoughts."
]
                formatted_resp = [response.format(movie_title) for response in resp]
                response = random.choice(formatted_resp)
                if self.llm_enabled:
                    return util.simple_llm_call(self.persona_prompt, response)
                return response
            elif (sentiment > 0):
                responses = [
                    "Okay, you liked {0}! What are your thoughts on another film?",
                    "Great to hear you enjoyed {0}! Got any other movies to discuss?",
                    "You appreciated {0}, right? Tell me about a different movie you've seen.",
                    "So, you're a fan of {0}! How about sharing your views on another title?",
                    "Loved {0}, didn't you? I'm curious about another movie you've watched!",
                    "You found {0} enjoyable! Can you mention another film and your thoughts on it?",
                    "It seems {0} was to your liking! What's another movie that caught your eye?",
                ]
                movie_title = '"' + movie_titles[0] + '"'
                response = random.choice(responses).format(movie_title)
                self.user_ratings[movie_index[0]] = 1
                self.num_inputted += 1
                if self.num_inputted == 5:
                    pass
                    #print(response)
                else: 
                    if self.llm_enabled:
                        return util.simple_llm_call(self.persona_prompt, response)
                    return response
            else:
                responses = [
                    "Okay, you did not like {0}! What did you like then?",
                    "Sorry to hear that you did not enjoy {0}! Got any other movies to discuss?",
                    "You did not appreciate {0}, right? Tell me about a different movie you like.",
                    "So, you're not a fan of {0}! How about sharing your views on another title?",
                    "You did not find {0} enjoyable! Can you mention another film and your thoughts on it?"
                ]
                movie_title = '"' + movie_titles[0] + '"'
                response = random.choice(responses).format(movie_title)
                self.user_ratings[movie_index[0]] = -1
                self.num_inputted += 1
                if self.num_inputted == 5:
                    pass
                    #print(response)
                else: 
                    return response
                
        if self.num_inputted >= 5: 
            #print("That's enough for me to make a recommendation.")
            
            #self.recommendations = self.recommend(self.user_ratings, self.ratings, 10, self.llm_enabled)
            #self.recommendations_given += 1
            #response_templates = [
            #    "I recommend {}. Interested in another suggestion?",
            #    "{} might be great movie recommendation for you. Would you like to hear about another one?",
            #    "Given what you told me, I think you would like {}. Shall I suggest another?",
            #    "You might enjoy {}. Want more recommendations?",
            #    "Let's go with {} for your recommendation this time. Another one?",
            #    "{} could be a great recommendation. Need another recommendation?",
            #    ]
            #selected_template = random.choice(response_templates)
            #formatted_response = selected_template.format('"' + self.titles[self.recommendations[0]][0] + '"')
            #self.recommending = True
            #return formatted_response
            self.recommendations = self.recommend(self.user_ratings, self.ratings, 10, self.llm_enabled)
            self.recommending = True
            return "That's enough for me to make a recommendation. Would you like a recommendation?"
            

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input: str):
        """
        Given an input text which has been pre-processed, this method uses an LLM to identify
        and return a list representing the emotions expressed within the text. The LLM is directed
        to consider the complexity of human emotions, including the possibility of mixed or nuanced
        expressions, and to return an exhaustive list of all relevant emotions identified.

        The method focuses on a set of primary emotions for simplicity: Anger, Disgust, Fear, Happiness,
        Sadness, and Surprise. It prompts the LLM to specifically consider these emotions, even in complex
        scenarios where multiple emotions might be present simultaneously.

        The LLM's response is expected to be a list of detected emotions based on the content of the preprocessed input.
        If no emotions are found, the method should return an empty list.

        :param preprocessed_input: a user-supplied line of text that has been pre-processed.
        :returns: a list of emotions detected in the text, or an empty list if no emotions are identified.
        Possible emotions include: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise".
        """
        if self.llm_enabled:
            system_prompt = """Your task is to analyze the given text and accurately identify the significant emotional expressions present, if any. It's essential to recognize that a text might express more than one emotions or no emotion at all. Focus on the clear and explicit expressions of emotions, avoiding assumptions where the emotion is not clearly stated. Identify key words or synonym of the emotion words.

Anger: Look for expressions of frustration or annoyance, e.g., 'I am quite frustrated by these awful recommendations!!!'
Happiness: Identify statements that express joy or satisfaction, e.g., 'Great suggestion! It put me in a great mood!'
Surprise: Detect expressions of astonishment or shock, e.g., 'Woah!!'
Sadness: Identify expressions of sorrow or melancholy, e.g., 'I am so sad. I just watched the saddest movie ever.'
Fear: Look for key words that express fear such as "I am so startled! That was frightening"
Disgust: Identify expressions of disgust such as "Disgusting!"
No Emotion: Recognize neutral statements without emotional content, e.g., 'What movie would you suggest I watch next?' implies no emotional expression. If there is no emotion, everything should be set to False.
For example, if the input 'Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they're pissing me off,' the detected emotions should be surprise and anger.

Mark the identified emotion as True and everything else as False.""" 
            message = preprocessed_input
            json_class = SentimentExtractor
            response = util.json_llm_call(system_prompt, message, json_class)
            emotions = set()
            for key in response:
                if response[key] == True:
                    emotions.add(key)
        return emotions


    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """

        titles = re.findall(r'"([^"]*)"', preprocessed_input)
        return titles

    def normalize_title(self, title):
        """Normalize title for comparison and extract year if present."""
        # Extract year if present and normalize title
        match = re.match(r'^(.*?)(?: \((\d{4})\))?$', title)
        if match:
            normalized_title = match.group(1).lower().strip()
            year = match.group(2)
            # Handle movies with articles moved to the end
            normalized_title = re.sub(r'^(a |an |the )(.*)', r'\2, \1', normalized_title).strip()
            return normalized_title, year
        else:
            return title.lower().strip(), None
        
    

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if self.llm_enabled:
            system_prompt = "Translate the user input into English. Remember to use the literal translation. For example, 'Jermand' is Danish so the appropriate output should be 'Iron Man.' If the output is 'Tote Männer Tragen Kein Plaid', return 'Dead Men Don't Wear Plaid'. The possible languages to translate from are German, Danish, Spanish, French, and Italian. Don't include any punctuation in your answer, and also don't mention which language it is translated from. Remember in Danish, 'en' refers to the."
            message = title
            json_class = Translator

            response = util.json_llm_call(system_prompt, message, json_class)
            normalized_title, year = self.normalize_title(response['Translation'])
            matching_indices = []
            for index, (movie_title, _) in enumerate(self.titles):
                db_title, db_year = self.normalize_title(movie_title)
                # Match title and year if provided, otherwise match title only
                if normalized_title == db_title and (year is None or year == db_year):
                    matching_indices.append(index)   
        
        else:
            normalized_title, year = self.normalize_title(title)
            matching_indices = []

            for index, (movie_title, _) in enumerate(self.titles):
                db_title, db_year = self.normalize_title(movie_title)
                # Match title and year if provided, otherwise match title only
                if normalized_title == db_title and (year is None or year == db_year):
                    matching_indices.append(index)

        return matching_indices

    def extract_sentiment(self, preprocessed_input):
        
        words = preprocessed_input.split()
        positive_count, negative_count = 0, 0
        negated = False  # Track whether we are currently in a negated context

        for i, word in enumerate(words):
            # Reset negation at clause boundaries
            if word in self.clause_boundaries:
                negated = False
                continue

            stemmed_word = self.stemmer.stem(word)
            
            new_sentiment = {self.stemmer.stem(key): value for key, value in self.sentiment.items()}
                

            # Determine sentiment value for sentiment-bearing words
            if stemmed_word in new_sentiment:  
                sentiment_val = 1 if new_sentiment[stemmed_word] == 'pos' else -1

                # Apply negation if active and then reset negation
                if negated:
                    sentiment_val *= -1
                    negated = False  # Reset negation after it's been applied

                # Update positive or negative count based on sentiment value
                if sentiment_val > 0:
                    positive_count += 1
                else:
                    negative_count += 1

            # Check and update negation status based on negation words
            if word in self.negation_words:
                negated = True

        # Determine overall sentiment based on counts
        if positive_count > negative_count:
            return 1  # Positive sentiment
        elif negative_count > positive_count:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment


    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        # Initialize the binarized ratings matrix with the same shape as ratings
        bin_ratings = np.zeros_like(ratings)

        # Apply the binarization logic
        bin_ratings[ratings > threshold] = 1
        bin_ratings[(ratings <= threshold) & (ratings > 0)] = -1
        # Ratings equal to 0 remain unchanged (implicitly handled by np.zeros_like)

        return bin_ratings


        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """
        norm = np.linalg.norm(ratings_matrix, axis=1)
        norm[norm == 0] = 1
        normalized_matrix = ratings_matrix / norm[:, np.newaxis]

        similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

        # Compute scores by multiplying the similarity matrix with the user's ratings
        # This operation identifies movies similar to those the user likes
        scores = np.dot(similarity_matrix, user_ratings)

        # Exclude movies the user has already rated
        # Set scores of rated movies to a large negative value to exclude them from recommendations
        scores[user_ratings > 0] = -np.inf

        # Get indices of movies with the highest scores
        recommended_movie_indices = np.argsort(scores)[::-1][:k]

        return recommended_movie_indices.tolist()


        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.
        
    
#         bin_user_ratings = self.binarize(user_ratings)
#         bin_ratings_matrix = self.binarize(ratings_matrix)
#         n_movies = ratings_matrix.shape[0]
        
#         sim_matrix = np.zeros((n_movies, n_movies))
        
#         for i in range(n_movies):
#             for j in range(i+1, n_movies):
#                 sim_matrix[i, j] = sim_matrix[j, i] = self.similarity(bin_user_ratings[i], bin_ratings_matrix[j])
        
#         scores = np.dot(sim_matrix, user_ratings)
#         recommendations = sorted(scores, reverse=True)
        
#         return recommendations[:k]




        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################


    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Hello! My name is MovieRobot. I am here to help you find your next favorite movie...
        To start, enter a list of movies you have already seen. Provide your ratings for each.
        Then based on your input I will generate a list of movies of highest cosine similarity
        to what you already prefer (add more here).
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
