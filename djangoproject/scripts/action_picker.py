from sentence_transformers import SentenceTransformer, util
from diffusers_api.models import Action

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similiarity(sentenceA, sentenceB):

    tmpA = [sentenceA]
    tmpB = [sentenceB]

    #Compute embedding for both lists
    embeddings1 = model.encode(tmpA, convert_to_tensor=True)
    embeddings2 = model.encode(tmpB, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores[0][0]

def pick_action(action_prompt):
    action_set = Action.objects.all()
    picked_action = action_set.first()
    max_score = calculate_similiarity(picked_action.description, action_prompt)
    for action in action_set:
        action_score = calculate_similiarity(action.description, action_prompt)
        if(action_score>max_score):
            picked_action = action
    
    return picked_action

