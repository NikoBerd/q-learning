import Foundation

// Define the grid position struct
struct GridPosition: Hashable {
    let row: Int
    let col: Int
}

// Define the grid world environment
let gridSize = 5
let numActions = 4 // Up, Down, Left, Right
let numEpisodes = 1000

// Initialize the Q-values with zeros
var qValues = [GridPosition: [Double]]()

// Define the rewards and obstacles in the grid world
let rewards: [[Double]] = [
    [-1, -1, -1, -1, 10],
    [-1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1],
    [-1, -1, -1, -1, -1]
]

let obstacles: Set<GridPosition> = [GridPosition(row: 1, col: 3), GridPosition(row: 2, col: 3), GridPosition(row: 3, col: 3)]

// Define the learning parameters
let learningRate: Double = 0.5
let discountFactor: Double = 0.8
let epsilon: Double = 0.2

// Function to choose an action based on epsilon-greedy policy
func chooseAction(state: GridPosition) -> Int {
    if Double.random(in: 0...1) < epsilon {
        // Randomly choose an action
        return Int.random(in: 0..<numActions)
    } else {
        // Choose the action with the highest Q-value
        let actionValues = qValues[state]!
        let maxQValue = actionValues.max()!
        return actionValues.firstIndex(of: maxQValue)!
    }
}

// Function to update Q-values based on the observed reward
func updateQValue(state: GridPosition, action: Int, nextState: GridPosition, reward: Double) {
    let currentQValue = qValues[state]![action]
    let maxNextQValue = qValues[nextState]!.max()!
    let tdError = reward + discountFactor * maxNextQValue - currentQValue
    qValues[state]![action] += learningRate * tdError
}

// Q-learning algorithm
func qLearning() {
    for _ in 0..<numEpisodes {
        // Randomly initialize the starting position
        var currentState = GridPosition(row: Int.random(in: 0..<gridSize), col: Int.random(in: 0..<gridSize))
        
        // Initialize the Q-values for the current state if necessary
        if qValues[currentState] == nil {
            qValues[currentState] = Array(repeating: 0.0, count: numActions)
        }
        
        while currentState != GridPosition(row: 0, col: gridSize - 1) {
            let action = chooseAction(state: currentState)
            
            // Perform the action and observe the next state and reward
            var nextState: GridPosition
            switch action {
            case 0: // Up
                nextState = GridPosition(row: currentState.row - 1, col: currentState.col)
            case 1: // Down
                nextState = GridPosition(row: currentState.row + 1, col: currentState.col)
            case 2: // Left
                nextState = GridPosition(row: currentState.row, col: currentState.col - 1)
            case 3: // Right
                nextState = GridPosition(row: currentState.row, col: currentState.col + 1)
            default:
                fatalError("Invalid action")
            }
            
            // Check for obstacles and boundary conditions
            if obstacles.contains(nextState) ||
                nextState.row < 0 || nextState.row >= gridSize ||
                nextState.col < 0 || nextState.col >= gridSize {
                nextState = currentState // Stay in the current state
            }
            
            // Initialize the Q-values for the next state if necessary
            if qValues[nextState] == nil {
                qValues[nextState] = Array(repeating: 0.0, count: numActions)
            }
            
            // Update the Q-value based on the observed reward
            let reward = rewards[nextState.row][nextState.col]
            updateQValue(state: currentState, action: action, nextState: nextState, reward: reward)
            
            currentState = nextState
        }
    }
}

// Run the Q-learning algorithm
qLearning()

// Print the learned Q-values
for row in 0..<gridSize {
    for col in 0..<gridSize {
        let position = GridPosition(row: row, col: col)
        print(qValues[position]!)
    }
    print()
}
