import SectionNAD from '../components/SectionNAD';
import SectionHandSpeak from '../components/SectionHandSpeak';
import SectionGallaudet from '../components/SectionGallaudet';

const Resources = () => {
    return (
    <div className="App">
      <header className="App-header">
        <h1>Resources</h1>
        <div className = 'container'>
          There are plenty of resources to learn more about ASL below!
          These are from trusted sources in the Deaf community and are 
          a great way to learn more about ASL from their perspectives. 
        </div>
      </header>

      <div className = 'Section-container'>
        <SectionNAD />
        <SectionHandSpeak />
        <SectionGallaudet />
      </div>
    </div>
    );
}

export default Resources;
