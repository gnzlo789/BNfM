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
	['breakfast', 'L', 1.0],
	['breakfast', 'M', 0.0],
	['breakfast', 'M', 0.0],
	['breakfast', 'J', 0.0],
	['breakfast', 'V', 0.0],
	['breakfast', 'S', 0.0],
	['breakfast', 'D', 0.0],

	['lunch', 'L', 0.0],
	['lunch', 'M', 0.5],
	['lunch', 'M', 0.5],
	['lunch', 'J', 0.0],
	['lunch', 'V', 0.0],
	['lunch', 'S', 0.0],
	['lunch', 'D', 0.0],

	['dinner', 'L', 1.0],
	['dinner', 'M', 0.0],
	['dinner', 'M', 0.0],
	['dinner', 'J', 0.0],
	['dinner', 'V', 0.0],
	['dinner', 'S', 0.0],
	['dinner', 'D', 0.0],
	
	['hangout', 'L', 1.0],
	['hangout', 'M', 0.0],
	['hangout', 'M', 0.0],
	['hangout', 'J', 0.0],
	['hangout', 'V', 0.0],
	['hangout', 'S', 0.0],
	['hangout', 'D', 0.0],

	['call', 'L', 1.0],
	['call', 'M', 0.0],
	['call', 'M', 0.0],
	['call', 'J', 0.0],
	['call', 'V', 0.0],
	['call', 'S', 0.0],
	['call', 'D', 0.0],

	['meeting', 'L', 1.0],
	['meeting', 'M', 0.0],
	['meeting', 'M', 0.0],
	['meeting', 'J', 0.0],
	['meeting', 'V', 0.0],
	['meeting', 'S', 0.0],
	['meeting', 'D', 0.0]
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
	['breakfast', 'S', '8:00', 0, 0.0], # PlaceId = 0 = Home
	['breakfast', 'L', '9:00', 1, 1.0], # PlaceId = 1 = Bar
	
	['lunch', 'V', '12:00', 1, 1.0],

	['dinner', 'V', '20:00', 1, 1.0],

	['hangout', 'V', '16:00', 1, 1.0],

	['call', 'V', '11:00', 1, 1.0],

	['meeting', 'V', '11:00', 1, 1.0],
], [meeting_type, day, time]) 

participant = ConditionalProbabilityTable([
	['L', '9:00', 1, True, 1.0], 

	['V', '9:30', 1, True, 1.0], 
	
	['V', '12:00', 1, True, 1.0],

	['V', '20:00', 1, True, 1.0],

	['V', '16:00', 1, True, 1.0],

	['V', '11:00', 1, False, 1.0],
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

print "== Meeting Type BEFORE =="
print meeting_type

print "== Add new key to meeting type =="
meeting_type.add_items( ['call'] )
meeting_type.train( ['call'], inertia=0.7 )

print "== Meeting Type AFTER =="
print meeting_type

print

print "== Duration BEFORE =="
print duration

print "== Add new row to the table =="
# I have to add every new possibility manually
duration.add_items( [['call', 30], ['call',60],['call', 120],['breakfast', 120], ['lunch', 120]] )
duration.train( [['call', 120], ['call', 30]] )

print "== Duration AFTER =="
print duration

print

print "== Network Training =="
#data = [ ['call', 30], ['breakfast', 120], ['lunch', 120] ]
data = [ ['breakfast', 30, 'L', '9:00', 1, True] ]
network.train( data )
print "== Meeting Type =="
print meeting_type
print "== Duration =="
print duration

network.bake()

# The second observation is that the meeting type is a 'meeting'
# second_observation = { 'type' : 'call' } # Works
second_observation = { 'duration' : 30 }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
beliefs = network.forward_backward( second_observation )

# Convert the beliefs into a more readable format
beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
