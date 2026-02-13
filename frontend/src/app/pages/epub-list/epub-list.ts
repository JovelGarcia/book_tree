import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-epub-list',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './epub-list.html',
})
export class EpubListComponent implements OnInit {

  epubs: any[] = [];

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get<any[]>('http://127.0.0.1:8000/api/epubs/')
      .subscribe(data => {
        this.epubs = data;
      });
  }
}
