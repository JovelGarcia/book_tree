import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MediaGraph } from './media-graph';

describe('MediaGraph', () => {
  let component: MediaGraph;
  let fixture: ComponentFixture<MediaGraph>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaGraph]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MediaGraph);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
