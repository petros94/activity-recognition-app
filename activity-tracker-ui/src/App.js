import logo from './logo.svg';
import './App.css';
import {ChakraProvider, Box, Heading, Text} from "@chakra-ui/react";
import Activity from "./Activity";

function App() {
  return (
      <ChakraProvider>
          <Box textAlign="center" py={10} px={6}>
              <Heading as="h2" size="xl" mt={6} mb={2}>
                  AI Activity Tracker
              </Heading>
              <Text color={'gray.500'}>
                  Track your everyday activity with our super advanced deep learning algorithm.
              </Text>
          </Box>
          <Activity/>
      </ChakraProvider>
  );
}

export default App;
