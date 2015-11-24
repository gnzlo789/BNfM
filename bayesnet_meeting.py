from pomegranate import *

# Meeting Type
meeting_type = DiscreteDistribution({
	'breakfast': 1./6,
	'lunch': 1./6,
	'dinner': 1./6,
	'hangout': 1./6,
	'call': 1./6,
	'meeting': 1./6
})

duration = ConditionalProbabilityTable([
	['breakfast', 30, 1.0],
	['breakfast', 60, 0.0],

	['lunch', 30, 0.5],
	['lunch', 60, 0.5],

	['dinner', 30, 0.0],
	['dinner', 60, 1.0],

	['hangout', 30, 0.0],
	['hangout', 60, 1.0],

	['call', 30, 0.0],
	['call', 60, 1.0],

	['meeting', 30, 0.0],
	['meeting', 60, 1.0],
], [meeting_type])


day = ConditionalProbabilityTable([
	['breakfast', 'Monday', 1.0],
	['breakfast', 'Tuesday', 0.0],
	['breakfast', 'Wednesday', 0.0],
	['breakfast', 'Thursday', 0.0],
	['breakfast', 'Friday', 0.0],
	['breakfast', 'Saturday', 0.0],
	['breakfast', 'Sunday', 0.0],

	['lunch', 'Monday', 0.0],
	['lunch', 'Tuesday', 0.5],
	['lunch', 'Wednesday', 0.5],
	['lunch', 'Thursday', 0.0],
	['lunch', 'Friday', 0.0],
	['lunch', 'Saturday', 0.0],
	['lunch', 'Sunday', 0.0],

	['dinner', 'Monday', 1.0],
	['dinner', 'Tuesday', 0.0],
	['dinner', 'Wednesday', 0.0],
	['dinner', 'Thursday', 0.0],
	['dinner', 'Friday', 0.0],
	['dinner', 'Saturday', 0.0],
	['dinner', 'Sunday', 0.0],

	['hangout', 'Monday', 1.0],
	['hangout', 'Tuesday', 0.0],
	['hangout', 'Wednesday', 0.0],
	['hangout', 'Thursday', 0.0],
	['hangout', 'Friday', 0.0],
	['hangout', 'Saturday', 0.0],
	['hangout', 'Sunday', 0.0],

	['call', 'Monday', 1.0],
	['call', 'Tuesday', 0.0],
	['call', 'Wednesday', 0.0],
	['call', 'Thursday', 0.0],
	['call', 'Friday', 0.0],
	['call', 'Saturday', 0.0],
	['call', 'Sunday', 0.0],

	['meeting', 'Monday', 1.0],
	['meeting', 'Tuesday', 0.0],
	['meeting', 'Wednesday', 0.0],
	['meeting', 'Thursday', 0.0],
	['meeting', 'Friday', 0.0],
	['meeting', 'Saturday', 0.0],
	['meeting', 'Sunday', 0.0]
], [meeting_type])

time = ConditionalProbabilityTable([
	['breakfast', '8:00', 0.3],
	['breakfast', '9:00', 0.5],
	['breakfast', '9:30', 0.2],

	['lunch', '12:00', 1.0],

	['dinner', '20:00', 1.0],

	['hangout', '16:00', 1.0],

	['call', '11:00', 1.0],

	['meeting', '11:00', 1.0],
], [meeting_type])

place = ConditionalProbabilityTable([
	['breakfast', 'Saturday', '8:00', 0, 0.0], # PlaceId = 0 = Home
	['breakfast', 'Monday', '9:00', 1, 1.0], # PlaceId = 1 = Bar

	['lunch', 'Friday', '12:00', 1, 1.0],

	['dinner', 'Friday', '20:00', 1, 1.0],

	['hangout', 'Friday', '16:00', 1, 1.0],

	['call', 'Friday', '11:00', 1, 1.0],

	['meeting', 'Friday', '11:00', 1, 1.0],
], [meeting_type, day, time])

participant = ConditionalProbabilityTable([
	['Monday', '9:00', 1, True, 1.0],

	['Friday', '9:30', 1, True, 1.0],

	['Friday', '12:00', 1, True, 1.0],

	['Friday', '20:00', 1, True, 1.0],

	['Friday', '16:00', 1, True, 1.0],

	['Friday', '11:00', 1, False, 1.0],
], [day, time, place])

# Make the states
s0 = State( meeting_type, name="type" )
s1 = State( duration, name="duration" )
s2 = State( day, name="day" )
s3 = State( time, name="time" )
s4 = State( place, name="place" )
s5 = State( participant, name="participant" )

# Make the bayes net, add the states, and the conditional dependencies.
network = BayesianNetwork( "BNfMS" )
network.add_states( [s0, s1, s2, s3, s4, s5] )
network.add_transition( s0, s1 )
network.add_transition( s0, s2 )
network.add_transition( s0, s3 )
network.add_transition( s0, s4 )
network.add_transition( s2, s4 )
network.add_transition( s3, s4 )
network.add_transition( s2, s5 )
network.add_transition( s3, s5 )
network.add_transition( s4, s5 )
network.bake()

# The first observation is that the duration is thirty minutes
first_observation = { 'duration' : 30 }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
beliefs = network.forward_backward( first_observation )

# Convert the beliefs into a more readable format
beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )


### Testing the new add_items method

# print "== Meeting Type BEFORE =="
# print meeting_type

# print "== Add new key to meeting type =="
meeting_type.add_items( ['another_type'] )
meeting_type.train( ['another_type'], inertia=0.7 )

# print "== Meeting Type AFTER =="
# print meeting_type

# print

# print "== Duration BEFORE =="
# print duration

print "== Add new row to the table =="
duration.add_items( [['dinner', 120]] )
data = [ ['breakfast', 120, 'Monday', '9:00', 1, True] ] # It is necessary to update the probabilities
network.update( data ) # Automatically generates the new possible values.

print "== Duration AFTER =="
print duration

print

print "== Network Training =="
data = [ ['call', 30, 'Monday', '9:00', 1, True], ['breakfast', 120, 'Monday', '9:00', 1, True] ]
network.train( data )
print "== Meeting Type =="
print meeting_type
print "== Duration =="
print duration
print "== Place =="
print place

# The second observation is that the meeting type is a 'meeting'
#second_observation = { 'type' : 'call' } # Works
second_observation = { 'type' : 'call', 'duration': 30 }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
beliefs = network.forward_backward( second_observation )

# Convert the beliefs into a more readable format
beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
