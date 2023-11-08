import Section from '../components/Section';

const Home = () => {
    return(
    <div className="App">
      <header className="App-header">
        <h1>Welcome to ASLingo</h1>
        <a>An Application to Learn ASL, American Sign Language</a>
      </header>
      
      <Section 
        header1 = 'Our Goal'
        header2 = 'The goal of ASLingo is to allow people all of all different abilities and skill levels learn ASL, American Sign Language. '
        header3 = 'We aim to do that by helping our users not only recognize signs done by someone else, but by allowing users to practice and check their form through the use of our ASLingo web application. ' 
        header4 = 'Go to the "Exercises" Page to get started with your first lesson!' 
        />

      <Section 
        header1 = 'Resources'
        header2 = 'We have many resources available for our users to learn, which were made with the help of those in the Deaf community.'
        header3 = 'There are many links to check out if you would like to learn more about the culture and community of Deaf or hard of hearing people.'
        header4 = 'Go to the "Resources" Page to learn more!'
      />

    </div>
    );
}

export default Home;