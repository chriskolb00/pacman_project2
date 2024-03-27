# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions
from util import manhattanDistance

# python pacman.py -p ReflexAgent -l testClassic
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***" 
        Foodz = newFood.asList()
        infinity = (float("inf"))
        
        ghostPos = successorGameState.getGhostPositions()
        fDistance = []
        gDistance = []
        for food in Foodz:
            fDistance.append(manhattanDistance(food, newPos)) 
        for ghost in ghostPos:
            gDistance.append(manhattanDistance(ghost, newPos))
        
        if currentGameState.getPacmanPosition() == newPos:
            return -infinity
        for dist in gDistance:
            if dist <2:
                return -infinity
        if len(fDistance)==0:
            return infinity
        
        return 99999/sum(fDistance) + 99999/len(fDistance)
        # return successorGameState.getScore()
        # python autograder.py -q q1 --no-graphics   
    
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minMax(gameState, index, depth=0):
            
            
            numIndex = gameState.getNumAgents()-1

            Act = None

            if gameState.isLose() or gameState.isWin() or (depth ==self.depth):
                return [self.evaluationFunction(gameState)]
            elif index == numIndex:
                depth +=1
                childIndex = self.index
            else:
                childIndex = index + 1
            # min
            if index != 0:
                min = float("inf")
                for action in gameState.getLegalActions(index):
                    succState = gameState.generateSuccessor(index, action)
                    newMin = minMax(succState, childIndex, depth)[0]
                    if newMin == min:
                        if bool(random.getrandbits(1)):
                            Act = action
                    elif newMin < min:
                        min = newMin
                        Act = action
                return min, Act
            # max
            else:
                max = -float("inf") 
                for action in gameState.getLegalActions(index):
                    succState = gameState.generateSuccessor(index, action)
                    newMax = minMax(succState, childIndex, depth)[0]
                    if newMax == max:
                        if bool(random.getrandbits(1)):
                            Act = action
                    elif newMax > max:
                        max = newMax
                        Act = action
                return max, Act 
        
        bestScore, bestMove = minMax(gameState, self.index)  
        return bestMove

        # util.raiseNotDefined()
        # python autograder.py -q q2 --no-graphics
        # python autograder.py -q q2

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_value(self, gameState, agentIndex, depth, a, b):
        depth+=1
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = -float('Inf')

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.min_value(successor, 1, depth, a, b))
            if v > b:
                return v
            a = max(a, v)

        return v

    def min_value(self, gameState, agentIndex, depth, a, b):
        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        v = float('Inf')

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents()-1:
                last_agent = True
            else:
                last_agent = False

            if last_agent:
                v = min(v, self.max_value(successor, 0, depth, a, b))
            else:
                v = min(v, self.min_value(successor, agentIndex+1, depth, a, b))

            if v < a:
                return v
            b = min(b, v)

        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        # alpha and beta 
        a = -float('Inf') 
        b = float('Inf')
        values = []
        agentIndex = self.index
        pacmanActions = gameState.getLegalActions(agentIndex) 
        print(pacmanActions)
        for act in pacmanActions:
            successor = gameState.generateSuccessor(agentIndex, act) 
            values.append((self.min_value(successor, 1, depth, a, b), act))
            #maxV = (maxEval, action)
            maxV = max(values)
            if maxV[0] > b:
                return maxV[1]
            a = max(a, maxV[0])

        return maxV[1]

        
        util.raiseNotDefined()
        # python autograder.py -q q3 --no-graphics

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
